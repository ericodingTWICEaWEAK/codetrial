#!/bin/sh

# Apply the formatters, or check that the tree is already where they would put
# it.
#
# One file list and one order, used by `make indent` through --write and by
# scripts/test.sh through --check, so what gets rewritten is what gets checked.
# scripts/git-pre-commit.sh mirrors these lists against the index in its own
# matcher language, and .claude/skills/codetrial-verify/SKILL.md describes them,
# so a lane added here needs adding in both.
#
# The order is not a preference. commentflow puts a blank line before a comment
# that sits inside a method chain, an array literal or an argument list, and
# `cargo fmt` takes that line back out; running the formatters last lets them
# have the final word. That is also why the check runs the whole chain against a
# copy of the tree rather than asking commentflow alone whether it would change
# something: on its own it always would, and the answer would be a gate nobody
# can satisfy.
#
# Style flags are deliberately absent. shfmt reads .editorconfig, which is where
# this repository's shell settings already live, and passing -i here would
# silently override the file for one caller.

set -u

mode=check
case "${1:-}" in
    "" | --check) ;;
    --write) mode=apply ;;
    *)
        echo "usage: ${0##*/} [--check | --write]" >&2
        exit 2
        ;;
esac

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT" || exit 1

# Untracked but non-ignored files are included, so a new script is covered
# before it is staged.
list()
{
    git ls-files --cached --others --exclude-standard -- "$@" | sort -u
}

rust=$(list 'build.rs' 'src/*.rs' 'tests/*.rs')

# The desktop app is a crate of its own, so `cargo fmt` at the root never sees
# it and it needs its manifest named.
desktop=$(list 'desktop/build.rs' 'desktop/src/*.rs' 'desktop/tests/*.rs')
shell=$(list '*.sh')

# tests/ as well as scripts/: the Python under tests/ drives the recording and
# calibration harnesses, and leaving it out of the formatter while `ruff check`
# reads it in the gate is a split nobody can keep track of.
python=$(list 'scripts/*.py' 'tests/*.py')

# The issue forms are in this list because Prettier is the only lane here that
# reads YAML: actionlint takes .github/workflows and stops. An unquoted colon in
# a description is a mapping where YAML wants a scalar, and GitHub answers the
# issue chooser with a parse error for it. Syntax only, so a mistyped key or an
# element `type` the schema does not have formats clean and still breaks the
# chooser: a green lane here is not a rendered form.
web=$(list '*.html' '*.js' '*.mjs' '*.cjs' \
    '.github/ISSUE_TEMPLATE/*.yml' '.github/ISSUE_TEMPLATE/*.yaml')
prettier=node_modules/.bin/prettier

have()
{
    command -v "$1" > /dev/null 2>&1 && return 0
    echo "${0##*/}: $1 not installed, skipping that lane${2:+ ($2)}" >&2
    return 1
}

# Run in $1, on the files named after it. Every lane is optional in the same way
# the rest of the gate treats an uninstalled tool: a contributor without it
# should still be able to run the half that works, and CI installs the ones that
# must not skip.
format()
{
    if [ -n "$rust$desktop$shell" ] && have commentflow; then
        # shellcheck disable=SC2086
        commentflow $rust $desktop $shell || return 1
    fi
    if [ -n "$rust" ] && have cargo; then
        cargo fmt || return 1
    fi
    if [ -n "$desktop" ] && have cargo; then
        cargo fmt --manifest-path desktop/Cargo.toml || return 1
    fi
    if [ -n "$python" ] && have ruff; then
        # shellcheck disable=SC2086
        ruff format --quiet $python || return 1
    fi
    if [ -n "$shell" ] && have shfmt; then
        # shellcheck disable=SC2086
        shfmt -w $shell || return 1
    fi
}

# Prettier shares no file with the chain above, so it needs neither the order
# nor the copy: the check reads the tree in place, which keeps the path stable
# enough for --cache to skip files unchanged since the last run, and it runs
# alongside the chain rather than after it. $1 is --write or --list-different.
pretty()
{
    [ -n "$web" ] && have "$prettier" "run npm ci" || return 0
    # The cache lives under target/ because `npm ci` empties node_modules.
    # shellcheck disable=SC2086
    "$prettier" --cache --cache-strategy content \
        --cache-location target/prettier-cache "$1" $web
}

if [ "$mode" = apply ]; then
    pretty --write > /dev/null &
    format
    chain=$?
    wait $! && [ "$chain" -eq 0 ]
    exit
fi

# The check writes too, just not here: the chain runs against a copy so that a
# dirty tree is never rewritten by a gate. Cargo.toml comes along because `cargo
# fmt` reads the manifest to find the crate, and .editorconfig because shfmt
# reads it to find the style.
work=$(mktemp -d) || exit 2
trap 'rm -rf "$work"' EXIT HUP INT TERM

files=$(printf '%s\n%s\n%s\n%s\n' "$rust" "$desktop" "$shell" "$python" | grep -v '^$')
manifests=Cargo.toml
[ -z "$desktop" ] || manifests="$manifests desktop/Cargo.toml"

# One tar pipe rather than a mkdir and a cp per file: 81 files was 162 process
# spawns and most of this gate's wall clock.
# shellcheck disable=SC2086
printf '%s\n' "$files" $manifests .editorconfig \
    | tar -cf - -T - \
    | (cd "$work" && tar -xf -) || exit 2

# A failed chain still waits for Prettier, so it does not outlive the script
# writing into a directory the trap has already removed. Exit status 1 from
# --list-different means drift, which the file list below reports.
pretty --list-different > "$work/.prettier" &
(cd "$work" && format)
chain=$?
wait $!
[ $? -le 1 ] && [ "$chain" -eq 0 ] || exit 2

drift=$(cat "$work/.prettier")
for file in $files; do
    cmp -s "$file" "$work/$file" || drift="$drift $file"
done

if [ -n "$drift" ]; then
    echo "Formatting drift; run 'make indent':" >&2
    for file in $drift; do
        echo "  $file" >&2
    done
    exit 1
fi

echo "Formatting clean."
