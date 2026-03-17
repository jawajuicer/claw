/**
 * AudioWorklet processor for real-time microphone capture.
 *
 * Captures audio from getUserMedia, downsamples from the native rate
 * (typically 44100/48000) to 16kHz, converts to Int16 PCM, and sends
 * 80ms chunks (1280 samples = 2560 bytes) to the main thread via
 * MessagePort for WebSocket transmission.
 *
 * All audio processing happens on the render thread — zero main-thread
 * blocking, zero GC pauses affecting capture quality.
 */
class ClawAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate;
    this.chunkSize = 1280; // 80ms at 16kHz
    this.outputBuffer = [];
    this.fractionalPos = 0;
    this.active = false;
    this.prevSample = 0;

    this.port.onmessage = (e) => {
      if (e.data.command === 'start') {
        this.active = true;
        this.outputBuffer = [];
        this.fractionalPos = 0;
        this.prevSample = 0;
      } else if (e.data.command === 'stop') {
        this.active = false;
        // Flush remaining buffer
        if (this.outputBuffer.length > 0) {
          this._sendChunk(this.outputBuffer.length);
          this.outputBuffer = [];
        }
      }
    };
  }

  process(inputs) {
    if (!this.active) return true;

    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0]; // Mono (first channel)

    // Resample using linear interpolation
    for (let i = 0; i < channelData.length; i++) {
      const currentSample = channelData[i];

      this.fractionalPos += 1;
      while (this.fractionalPos >= this.ratio) {
        this.fractionalPos -= this.ratio;

        // Linear interpolation between previous and current sample
        const frac = this.fractionalPos / this.ratio;
        const interpolated = this.prevSample + (currentSample - this.prevSample) * (1 - frac);
        this.outputBuffer.push(interpolated);

        if (this.outputBuffer.length >= this.chunkSize) {
          this._sendChunk(this.chunkSize);
          this.outputBuffer = [];
        }
      }

      this.prevSample = currentSample;
    }

    return true;
  }

  _sendChunk(length) {
    const int16 = new Int16Array(length);
    for (let i = 0; i < length; i++) {
      const s = this.outputBuffer[i] || 0;
      int16[i] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
    }
    this.port.postMessage({ type: 'audio', buffer: int16.buffer }, [int16.buffer]);
  }
}

registerProcessor('claw-audio-processor', ClawAudioProcessor);
