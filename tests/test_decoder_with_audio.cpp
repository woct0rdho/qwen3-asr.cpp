#include "text_decoder.h"

#include <cstdio>
#include <vector>
#include <fstream>
#include <algorithm>

static bool load_npy_f32(const std::string & path, std::vector<float> & data, 
                         std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return false; }
    
    uint8_t major, minor;
    if (fread(&major, 1, 1, f) != 1 || fread(&minor, 1, 1, f) != 1) { fclose(f); return false; }
    
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16;
        if (fread(&len16, 2, 1, f) != 1) { fclose(f); return false; }
        header_len = len16;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) { fclose(f); return false; }
    }
    
    std::vector<char> header(header_len + 1);
    if (fread(header.data(), 1, header_len, f) != header_len) { fclose(f); return false; }
    header[header_len] = '\0';
    
    std::string header_str(header.data());
    
    size_t shape_start = header_str.find("'shape': (");
    if (shape_start == std::string::npos) { fclose(f); return false; }
    shape_start += 10;
    
    size_t shape_end = header_str.find(")", shape_start);
    if (shape_end == std::string::npos) { fclose(f); return false; }
    
    std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);
    
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        
        int64_t dim = 0;
        while (pos < shape_str.size() && shape_str[pos] >= '0' && shape_str[pos] <= '9') {
            dim = dim * 10 + (shape_str[pos] - '0');
            pos++;
        }
        shape.push_back(dim);
    }
    
    int64_t total_elements = 1;
    for (auto d : shape) total_elements *= d;
    
    data.resize(total_elements);
    if (fread(data.data(), sizeof(float), total_elements, f) != (size_t)total_elements) {
        fclose(f); return false;
    }
    
    fclose(f);
    return true;
}

int main() {
    printf("=== Test Decoder with Reference Audio Features ===\n\n");
    
    // Load reference audio features
    std::vector<float> audio_features;
    std::vector<int64_t> audio_shape;
    if (!load_npy_f32("tests/reference/audio_features.npy", audio_features, audio_shape)) {
        fprintf(stderr, "Failed to load audio features\n");
        return 1;
    }
    printf("Audio features shape: [%lld, %lld]\n", (long long)audio_shape[0], (long long)audio_shape[1]);
    int32_t n_audio_frames = audio_shape[0];
    (void)audio_shape[1];
    
    // Load decoder
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("Model loaded\n");
    
    if (!decoder.init_kv_cache(2048)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    // Build token sequence (same as qwen3_asr.cpp)
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    const int32_t audio_start = 151669;
    const int32_t audio_pad = 151676;
    const int32_t audio_end = 151670;
    
    std::vector<int32_t> tokens;
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    tokens.push_back(audio_start);
    for (int i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(audio_pad);
    }
    tokens.push_back(audio_end);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    printf("Token sequence length: %zu\n", tokens.size());
    printf("audio_start_pos = 9\n");
    
    // Run forward pass with audio
    std::vector<float> logits;
    int32_t audio_start_pos = 9;
    
    if (!decoder.forward_with_audio(
            tokens.data(), tokens.size(),
            audio_features.data(), n_audio_frames,
            audio_start_pos, 0, logits)) {
        fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    printf("Logits size: %zu\n", logits.size());
    
    // Get last position logits
    const auto & cfg = decoder.get_config();
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_tokens = tokens.size();
    const float * last_logits = logits.data() + (n_tokens - 1) * vocab_size;
    
    // Find top 5
    std::vector<std::pair<float, int32_t>> top;
    for (int32_t i = 0; i < vocab_size; ++i) {
        top.push_back({last_logits[i], i});
    }
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    printf("\nTop 5 logits at last position:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  [%d] token=%d logit=%f\n", i, top[i].second, top[i].first);
    }
    
    printf("\nEOS token (%d) logit: %f\n", cfg.eos_token_id, last_logits[cfg.eos_token_id]);
    
    // Expected: token 11528 ("language") should be top, not EOS
    if (top[0].second == 11528) {
        printf("\nTEST PASSED: Top token is 11528 (language)\n");
    } else if (top[0].second == cfg.eos_token_id) {
        printf("\nTEST FAILED: Top token is EOS (%d)\n", cfg.eos_token_id);
    } else {
        printf("\nTEST RESULT: Top token is %d (expected 11528)\n", top[0].second);
    }
    
    return 0;
}
