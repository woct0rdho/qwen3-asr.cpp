#include <iostream>
#include <qwen3_asr.h>
#include <forced_aligner.h>

using namespace std;

void info(string text) {
    cerr << text << endl;
}

int main() {
    /*
    curl -L -o assets/qwen3-asr-0.6b-q8_0.gguf https://huggingface.co/Jaffe2718/Qwen3-ASR-GGUF/resolve/main/qwen3-asr-0.6b-q8_0.gguf?download=true
    curl -L -o assets/qwen3-forcedaligner-0.6b-f16.gguf https://huggingface.co/Jaffe2718/Qwen3-ASR-GGUF/resolve/main/qwen3-forcedaligner-0.6b-f16.gguf?download=true
    curl -L -o assets/qwen3-asr-1.7b-q4_1.gguf https://huggingface.co/Jaffe2718/Qwen3-ASR-GGUF/resolve/main/qwen3-asr-1.7b-q4_1.gguf?download=true
    curl -L -o assets/jfk.wav https://raw.githubusercontent.com/ggml-org/whisper.cpp/refs/heads/master/samples/jfk.wav
     */
    auto* asr = new qwen3_asr::Qwen3ASR();
    asr->load_model("assets/qwen3-asr-0.6b-q8_0.gguf");
    info("qwen3-asr-0.6b-q8_0.gguf loaded");
    qwen3_asr::transcribe_result result = asr->transcribe("assets/jfk.wav");
    info(result.text);
    info("-----------------");
    delete asr;

    auto* asr1_7 = new qwen3_asr::Qwen3ASR();
    asr1_7->load_model("assets/qwen3-asr-1.7b-q4_1.gguf");
    info("qwen3-asr-1.7b-q4_1.gguf loaded");
    qwen3_asr::transcribe_result result1_7 = asr1_7->transcribe("assets/jfk.wav");
    info(result1_7.text);
    info("-----------------");
    delete asr1_7;

    auto* alinger = new qwen3_asr::ForcedAligner();
    alinger->load_model("assets/qwen3-forcedaligner-0.6b-f16.gguf");
    info("qwen3-forcedaligner-0.6b-f16.gguf loaded");
    qwen3_asr::alignment_result align_result = alinger->align("assets/jfk.wav", result.text);
    for (auto & word : align_result.words) {
        cerr << "`" << word.word << "` (" << word.start << " --> " << word.end << ")" << endl;
    }
    info("-----------------");
    delete alinger;
    return 0;
}