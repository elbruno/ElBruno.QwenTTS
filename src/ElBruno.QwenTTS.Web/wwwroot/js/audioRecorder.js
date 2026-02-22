// Audio recorder for voice cloning — records microphone, resamples to 24 kHz mono 16-bit PCM WAV

let mediaRecorder = null;
let audioChunks = [];
let startTime = 0;

window.audioRecorder = {
    // Start recording from the microphone
    startRecording: async function () {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        startTime = Date.now();

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.start(100); // collect chunks every 100ms
        return true;
    },

    // Stop recording and return base64-encoded 24kHz mono 16-bit PCM WAV
    stopRecording: async function () {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') return null;

        return new Promise((resolve, reject) => {
            mediaRecorder.onstop = async () => {
                try {
                    // Stop all tracks
                    mediaRecorder.stream.getTracks().forEach(t => t.stop());

                    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioCtx = new AudioContext();
                    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
                    audioCtx.close();

                    // Resample to 24000 Hz mono
                    const targetSampleRate = 24000;
                    const numSamples = Math.round(decoded.duration * targetSampleRate);
                    const offlineCtx = new OfflineAudioContext(1, numSamples, targetSampleRate);
                    const source = offlineCtx.createBufferSource();
                    source.buffer = decoded;
                    source.connect(offlineCtx.destination);
                    source.start(0);
                    const resampled = await offlineCtx.startRendering();

                    // Encode as 16-bit PCM WAV
                    const samples = resampled.getChannelData(0);
                    const wavBytes = encodeWav(samples, targetSampleRate);
                    const base64 = arrayBufferToBase64(wavBytes);
                    resolve(base64);
                } catch (err) {
                    console.error('audioRecorder.stopRecording error:', err);
                    reject(err);
                }
            };
            mediaRecorder.stop();
        });
    },

    // Get recording duration in seconds
    getDuration: function () {
        if (!startTime || !mediaRecorder || mediaRecorder.state === 'inactive') return 0;
        return (Date.now() - startTime) / 1000;
    },

    // Check if currently recording
    isRecording: function () {
        return mediaRecorder && mediaRecorder.state === 'recording';
    }
};

function encodeWav(samples, sampleRate) {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);          // chunk size
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, sampleRate, true);   // sample rate
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);           // block align
    view.setUint16(34, 16, true);          // bits per sample

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, numSamples * 2, true);

    // Convert float32 [-1, 1] to int16
    for (let i = 0; i < numSamples; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return buffer;
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}
