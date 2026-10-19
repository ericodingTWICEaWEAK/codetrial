/// Whether playback is blocked after `observation`, given that it was
/// `blocked` before. Anything that says nothing about autoplay keeps it.
export function playbackBlocked(blocked, observation) {
  if (observation.type === "status") {
    if (typeof observation.allowed === "boolean") return !observation.allowed;
  } else if (observation.type === "played") {
    return false;
  } else if (
    observation.type === "rejected" &&
    observation.name === "NotAllowedError"
  ) {
    return true;
  }
  return blocked;
}

/// Room events and play promises can outlive the audio source they describe.
/// A new attachment or room invalidates older promise results. Room status
/// does not: the page's own play may succeed after LiveKit reported a block.
export function createAudioPlayback({
  notice,
  button,
  focusTarget,
  announcement,
}) {
  let room = null;
  let event = null;
  let element = null;
  let revision = 0;
  let pending = false;
  let blocked = false;

  function render() {
    if (!blocked && button.ownerDocument.activeElement === button)
      focusTarget.focus({ preventScroll: true });
    notice.hidden = !blocked;
    button.setAttribute("aria-disabled", String(pending));
    button.setAttribute("aria-busy", String(pending));
  }

  function observe(observation) {
    const next = playbackBlocked(blocked, observation);
    if (next !== blocked)
      announcement.textContent = next
        ? "Your browser blocked audio playback. Resume audio to hear the interviewer."
        : "Audio restored.";
    blocked = next;
    render();
  }

  function status() {
    if (element) observe({ type: "status", allowed: room?.canPlaybackAudio });
  }

  /// Drops the current source silently: a source that went away was not
  /// restored, so nothing is announced.
  function clear() {
    revision += 1;
    element = null;
    pending = false;
    blocked = false;
    announcement.textContent = "";
    render();
  }

  function setRoom(nextRoom, nextEvent) {
    if (room) room.off(event, status);
    room = nextRoom;
    event = nextEvent;
    clear();
    if (room) room.on(event, status);
  }

  /// Tears down only if `oldRoom` is still the current one, so a late event
  /// from a room already replaced cannot clear the new room's state.
  function release(oldRoom) {
    if (room === oldRoom) setRoom(null);
  }

  function play(nextElement) {
    if (!room) return Promise.resolve();
    revision += 1;
    element = nextElement;
    pending = false;
    render();
    if (room.canPlaybackAudio === false)
      observe({ type: "status", allowed: false });
    const current = revision;
    return Promise.resolve()
      .then(() => nextElement.play())
      .then(
        () => {
          if (current === revision) observe({ type: "played" });
        },
        (error) => {
          if (current === revision)
            observe({ type: "rejected", name: error?.name });
        },
      );
  }

  function forget(oldElement) {
    if (element === oldElement) clear();
  }

  function resume() {
    if (pending || !room || !element || !blocked) return;
    // A new source or room bumps the revision and makes this result stale.
    const current = revision;
    pending = true;
    render();
    // Both calls start inside the click gesture. Awaiting startAudio first
    // can spend transient activation before the element's play request. The
    // executor runs synchronously, and turns a throw into a rejection.
    const start = (call) => new Promise((resolve) => resolve(call()));
    void Promise.allSettled([
      start(() => room.startAudio()),
      start(() => element.play()),
    ]).then(([, played]) => {
      if (current !== revision) return;
      pending = false;
      observe(
        played.status === "fulfilled"
          ? { type: "played" }
          : { type: "rejected", name: played.reason?.name },
      );
    });
  }

  button.addEventListener("click", resume);
  render();
  return { setRoom, release, play, forget };
}
