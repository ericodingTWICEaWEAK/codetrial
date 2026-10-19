import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_RUNTIME_CONFIG,
  FAKE_SPEECH_CAPTURE,
  launchChromium,
  startStaticServer,
} from "./source.js";

let browser, server, base;
before(async () => {
  browser = await launchChromium({
    args: [
      "--autoplay-policy=document-user-activation-required",
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      FAKE_SPEECH_CAPTURE,
    ],
  });
  if (browser)
    ({ server, base } = await startStaticServer({
      runtimeConfig: DEFAULT_RUNTIME_CONFIG,
    }));
});
after(async () => {
  await browser?.close();
  await new Promise((resolve) => (server ? server.close(resolve) : resolve()));
});

test("real autoplay denial is visible and a gesture starts native audio", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await browser.newPage();
  t.after(() => page.close());
  // page.evaluate grants synthetic user activation. Run setup as a page
  // module instead, so the first play really occurs before any gesture.
  await page.route("**/playback-probe", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: `<span id="anchor" tabindex="-1">Interview</span>
        <div id="notice" hidden><p role="status">Audio blocked</p>
        <button id="resume">Resume audio</button></div>
        <span id="announcement" role="status"></span>
        <script type="module">
          import { createAudioPlayback } from '/audio-playback.js';
          const room = new EventTarget();
          room.on = room.addEventListener.bind(room);
          room.off = room.removeEventListener.bind(room);
          room.startAudio = async () => {};
          const playback = createAudioPlayback({
            notice: document.querySelector('#notice'),
            button: document.querySelector('#resume'),
            focusTarget: document.querySelector('#anchor'),
            announcement: document.querySelector('#announcement'),
          });
          playback.setRoom(room, 'playback');
          window.audio = new Audio('/audio/test-voice.wav');
          document.body.append(window.audio);
          await playback.play(window.audio);
          window.playbackReady = true;
        </script>`,
    }),
  );
  await page.goto(`${base}/playback-probe`);
  await page.waitForFunction(() => window.playbackReady);
  assert.equal(await page.locator("#notice").isVisible(), true);
  assert.equal(await page.evaluate(() => window.audio.paused), true);
  await page.locator("#resume").click();
  await page.waitForFunction(
    () => !window.audio.paused && document.querySelector("#notice").hidden,
  );
  assert.equal(await page.evaluate(() => document.activeElement.id), "anchor");
});

test("interview reconnect wiring recovers blocked Jim audio and clears on disconnect", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await browser.newPage();
  t.after(() => page.close());
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    const nativePlay = HTMLMediaElement.prototype.play;
    window.blockJim = true;
    HTMLMediaElement.prototype.play = function () {
      return this.id === "jim-audio" && window.blockJim
        ? Promise.reject(
            new DOMException("Test autoplay denial", "NotAllowedError"),
          )
        : nativePlay.call(this);
    };
  });
  await page.route("**/vendor/livekit-client.js", (route) =>
    route.fulfill({
      contentType: "text/javascript",
      body: `
      class Room {
        constructor() {
          window.testRoom = this;
          this.listeners = new Map();
          this.canPlaybackAudio = true;
          this.remoteParticipants = new Map();
          this.localParticipant = { identity: 'candidate', publishTrack: async () => {}, publishData: async () => {}, setMicrophoneEnabled: async () => {} };
        }
        on(event, handler) { const list = this.listeners.get(event) || []; list.push(handler); this.listeners.set(event, list); return this; }
        off(event, handler) { this.listeners.set(event, (this.listeners.get(event) || []).filter(h => h !== handler)); return this; }
        emit(event, ...args) { for (const handler of this.listeners.get(event) || []) handler(...args); }
        async connect() {}
        async disconnect() { this.emit('Disconnected'); }
        registerTextStreamHandler() {}
        async startAudio() { this.starts = (this.starts || 0) + 1; window.blockJim = false; }
      }
      const RoomEvent = Object.fromEntries(['DataReceived','ParticipantAttributesChanged','ParticipantConnected','ParticipantDisconnected','TrackSubscribed','TrackUnsubscribed','Reconnecting','Reconnected','Disconnected','AudioPlaybackStatusChanged'].map(name => [name, name]));
      window.LivekitClient = { Room, RoomEvent, Track: {Source: {Microphone: 'mic', Camera: 'camera'}} };
    `,
    }),
  );
  await page.route("**/api/token", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        token: "synthetic",
        serverUrl: "wss://invalid.test",
        durationMin: 45,
      }),
    }),
  );
  await page.goto(`${base}/interview.html?problem=chargeback-pair-match`);
  await page.locator("#camera-skip").click();
  await page.locator("#audio-heard").click();
  await page.waitForFunction(
    () => !document.querySelector("#audio-check-join").disabled,
  );
  await page.locator("#audio-check-join").click();
  await page.waitForFunction(
    () =>
      window.testRoom &&
      document.querySelector("#mic").textContent === "Mic on",
  );
  await page.evaluate(() => {
    const room = window.testRoom;
    const participant = {
      identity: "interviewer-jim",
      attributes: { "codetrial.agent_state": "listening" },
    };
    room.remoteParticipants.set(participant.identity, participant);
    room.emit("Reconnecting");
    room.emit("Reconnected");
    window.testTrack = {
      kind: "audio",
      attach: (audio) => {
        audio.src = "/audio/test-voice.wav";
      },
      detach: () => [document.querySelector("#jim-audio")],
    };
    room.emit("TrackSubscribed", window.testTrack, {}, participant);
  });
  await page.waitForSelector("#audio-playback-notice:not([hidden])");
  assert.equal(await page.locator("#jim-stage #resume-audio").count(), 1);
  await page.locator("#resume-audio").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#audio-playback-notice").hidden &&
      !document.querySelector("#jim-audio").paused,
  );
  assert.equal(await page.evaluate(() => window.testRoom.starts), 1);
  assert.equal(
    await page.evaluate(() => document.activeElement.id),
    "jim-stage",
  );
  // The wait for the report makes the sidebar inert while Jim may still be
  // speaking. Resume has to stay clickable over that overlay.
  await page.evaluate(() => {
    document.querySelector("#ending-overlay").hidden = false;
    window.testRoom.canPlaybackAudio = false;
    window.testRoom.emit("AudioPlaybackStatusChanged", false);
  });
  await page.waitForFunction(
    () => document.querySelector("#interview-sidebar").inert,
  );
  await page.locator("#resume-audio").click();
  await page.waitForSelector("#audio-playback-notice", { state: "hidden" });
  assert.equal(await page.evaluate(() => window.testRoom.starts), 2);
  await page.evaluate(() => {
    document.querySelector("#ending-overlay").hidden = true;
  });
  // Resumed from the keyboard, focus moves to the stage and stays visible.
  await page.evaluate(() => {
    window.testRoom.canPlaybackAudio = false;
    window.testRoom.emit("AudioPlaybackStatusChanged", false);
  });
  await page.waitForSelector("#audio-playback-notice:not([hidden])");
  await page.locator("#resume-audio").focus();
  await page.keyboard.press("Enter");
  await page.waitForSelector("#audio-playback-notice", { state: "hidden" });
  assert.deepEqual(
    await page.evaluate(() => [
      document.activeElement.id,
      getComputedStyle(document.activeElement).outlineStyle !== "none",
    ]),
    ["jim-stage", true],
  );
  await page.evaluate(() => {
    window.testRoom.canPlaybackAudio = false;
    window.testRoom.emit("AudioPlaybackStatusChanged", false);
  });
  await page.waitForSelector("#audio-playback-notice:not([hidden])");
  await page.evaluate(() => window.testRoom.emit("Disconnected"));
  await page.waitForSelector("#audio-playback-notice", { state: "hidden" });
  assert.deepEqual(errors, []);
});
