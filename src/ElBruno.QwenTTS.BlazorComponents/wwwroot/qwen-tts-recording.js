window.qwenTtsRecording = (() => {
    let audioContext;
    let source;
    let processor;
    let stream;
    let chunks = [];

    const unavailable = () => !navigator.mediaDevices?.getUserMedia || !window.AudioContext;

    const makeWav = (samples, sampleRate) => {
        const dataLength = samples.length * 2;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);
        const write = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
        write(0, "RIFF");
        view.setUint32(4, 36 + dataLength, true);
        write(8, "WAVEfmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        write(36, "data");
        view.setUint32(40, dataLength, true);
        samples.forEach((sample, index) => view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, sample)) * 0x7fff, true));
        return new Uint8Array(buffer);
    };

    return {
        isAvailable: () => !unavailable(),
        start: async () => {
            if (unavailable()) {
                throw new Error("Browser recording is unavailable.");
            }

            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new AudioContext();
            source = audioContext.createMediaStreamSource(stream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);
            chunks = [];
            processor.onaudioprocess = event => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
            source.connect(processor);
            processor.connect(audioContext.destination);
        },
        stop: async maxBytes => {
            if (!processor || !audioContext) {
                throw new Error("No active recording.");
            }

            const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
            const samples = new Float32Array(length);
            let offset = 0;
            chunks.forEach(chunk => {
                samples.set(chunk, offset);
                offset += chunk.length;
            });
            const bytes = makeWav(samples, audioContext.sampleRate);
            await window.qwenTtsRecording.dispose();
            if (bytes.byteLength > maxBytes) {
                throw new Error(`Recorded audio exceeds the ${maxBytes} byte limit.`);
            }
            let binary = "";
            bytes.forEach(value => binary += String.fromCharCode(value));
            return {
                fileName: "recording.wav",
                contentType: "audio/wav",
                dataBase64: btoa(binary)
            };
        },
        dispose: async () => {
            processor?.disconnect();
            source?.disconnect();
            stream?.getTracks().forEach(track => track.stop());
            await audioContext?.close();
            audioContext = undefined;
            source = undefined;
            processor = undefined;
            stream = undefined;
            chunks = [];
        }
    };
})();
