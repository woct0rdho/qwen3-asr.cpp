#include "audio_injection.h"

#include <cstdio>
#include <cmath>
#include <cassert>

using namespace qwen3_asr;

static bool float_eq(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) < eps;
}

static void test_find_audio_positions() {
    printf("Testing find_audio_positions...\n");
    
    int32_t input_ids[] = {151669, 151676, 151676, 151676, 151670};
    int32_t n_tokens = 5;
    int32_t audio_pad_token_id = 151676;
    
    auto positions = find_audio_positions(input_ids, n_tokens, audio_pad_token_id);
    
    assert(positions.size() == 3);
    assert(positions[0] == 1);
    assert(positions[1] == 2);
    assert(positions[2] == 3);
    
    printf("  PASS: Found %zu audio positions at [%d, %d, %d]\n", 
           positions.size(), positions[0], positions[1], positions[2]);
}

static void test_embed_tokens() {
    printf("Testing embed_tokens...\n");
    
    const int32_t vocab_size = 10;
    const int32_t hidden_size = 4;
    
    float token_embd[vocab_size * hidden_size];
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            token_embd[i * hidden_size + j] = static_cast<float>(i * 10 + j);
        }
    }
    
    int32_t input_ids[] = {0, 5, 3};
    const int32_t n_tokens = 3;
    
    float output[n_tokens * hidden_size];
    embed_tokens(input_ids, n_tokens, token_embd, vocab_size, hidden_size, output);
    
    assert(float_eq(output[0], 0.0f));
    assert(float_eq(output[1], 1.0f));
    assert(float_eq(output[2], 2.0f));
    assert(float_eq(output[3], 3.0f));
    
    assert(float_eq(output[4], 50.0f));
    assert(float_eq(output[5], 51.0f));
    assert(float_eq(output[6], 52.0f));
    assert(float_eq(output[7], 53.0f));
    
    assert(float_eq(output[8], 30.0f));
    assert(float_eq(output[9], 31.0f));
    assert(float_eq(output[10], 32.0f));
    assert(float_eq(output[11], 33.0f));
    
    printf("  PASS: Token embeddings correctly looked up\n");
}

static void test_inject_audio_embeddings() {
    printf("Testing inject_audio_embeddings...\n");
    
    const int32_t n_tokens = 5;
    const int32_t hidden_size = 4;
    const int32_t n_audio_frames = 3;
    
    float token_embeddings[n_tokens * hidden_size];
    for (int i = 0; i < n_tokens * hidden_size; ++i) {
        token_embeddings[i] = static_cast<float>(i);
    }
    
    float audio_features[n_audio_frames * hidden_size];
    for (int i = 0; i < n_audio_frames * hidden_size; ++i) {
        audio_features[i] = static_cast<float>(100 + i);
    }
    
    std::vector<int32_t> audio_positions = {1, 2, 3};
    
    bool success = inject_audio_embeddings(
        token_embeddings, n_tokens, hidden_size,
        audio_features, n_audio_frames, audio_positions);
    
    assert(success);
    
    assert(float_eq(token_embeddings[0], 0.0f));
    assert(float_eq(token_embeddings[1], 1.0f));
    assert(float_eq(token_embeddings[2], 2.0f));
    assert(float_eq(token_embeddings[3], 3.0f));
    
    assert(float_eq(token_embeddings[4], 100.0f));
    assert(float_eq(token_embeddings[5], 101.0f));
    assert(float_eq(token_embeddings[6], 102.0f));
    assert(float_eq(token_embeddings[7], 103.0f));
    
    assert(float_eq(token_embeddings[8], 104.0f));
    assert(float_eq(token_embeddings[9], 105.0f));
    assert(float_eq(token_embeddings[10], 106.0f));
    assert(float_eq(token_embeddings[11], 107.0f));
    
    assert(float_eq(token_embeddings[12], 108.0f));
    assert(float_eq(token_embeddings[13], 109.0f));
    assert(float_eq(token_embeddings[14], 110.0f));
    assert(float_eq(token_embeddings[15], 111.0f));
    
    assert(float_eq(token_embeddings[16], 16.0f));
    assert(float_eq(token_embeddings[17], 17.0f));
    assert(float_eq(token_embeddings[18], 18.0f));
    assert(float_eq(token_embeddings[19], 19.0f));
    
    printf("  PASS: Audio embeddings correctly injected at positions 1, 2, 3\n");
}

static void test_inject_audio_full() {
    printf("Testing inject_audio (full pipeline)...\n");
    
    const int32_t vocab_size = 200000;
    const int32_t hidden_size = 1024;
    const int32_t n_tokens = 5;
    const int32_t n_audio_frames = 3;
    
    std::vector<float> token_embd(vocab_size * hidden_size);
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            token_embd[i * hidden_size + j] = static_cast<float>(i % 1000) + static_cast<float>(j) / 10000.0f;
        }
    }
    
    int32_t input_ids[] = {151669, 151676, 151676, 151676, 151670};
    
    std::vector<float> audio_features(n_audio_frames * hidden_size);
    for (int i = 0; i < n_audio_frames; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            audio_features[i * hidden_size + j] = static_cast<float>(500 + i) + static_cast<float>(j) / 10000.0f;
        }
    }
    
    audio_injection_context ctx;
    ctx.token_embd = token_embd.data();
    ctx.vocab_size = vocab_size;
    ctx.hidden_size = hidden_size;
    ctx.tokens.audio_start_token_id = 151669;
    ctx.tokens.audio_end_token_id = 151670;
    ctx.tokens.audio_pad_token_id = 151676;
    
    injection_result result = inject_audio(
        input_ids, n_tokens,
        audio_features.data(), n_audio_frames,
        ctx);
    
    assert(result.success);
    assert(result.seq_len == n_tokens);
    assert(result.hidden_size == hidden_size);
    assert(result.embeddings.size() == static_cast<size_t>(n_tokens * hidden_size));
    
    float expected_start = static_cast<float>(151669 % 1000);
    assert(float_eq(result.embeddings[0], expected_start, 0.01f));
    
    float expected_audio_0 = 500.0f;
    assert(float_eq(result.embeddings[1 * hidden_size], expected_audio_0, 0.01f));
    
    float expected_audio_1 = 501.0f;
    assert(float_eq(result.embeddings[2 * hidden_size], expected_audio_1, 0.01f));
    
    float expected_audio_2 = 502.0f;
    assert(float_eq(result.embeddings[3 * hidden_size], expected_audio_2, 0.01f));
    
    float expected_end = static_cast<float>(151670 % 1000);
    assert(float_eq(result.embeddings[4 * hidden_size], expected_end, 0.01f));
    
    printf("  PASS: Full injection pipeline works correctly\n");
    printf("    - Position 0 (audio_start): %.2f (expected %.2f)\n", 
           result.embeddings[0], expected_start);
    printf("    - Position 1 (audio frame 0): %.2f (expected %.2f)\n", 
           result.embeddings[1 * hidden_size], expected_audio_0);
    printf("    - Position 2 (audio frame 1): %.2f (expected %.2f)\n", 
           result.embeddings[2 * hidden_size], expected_audio_1);
    printf("    - Position 3 (audio frame 2): %.2f (expected %.2f)\n", 
           result.embeddings[3 * hidden_size], expected_audio_2);
    printf("    - Position 4 (audio_end): %.2f (expected %.2f)\n", 
           result.embeddings[4 * hidden_size], expected_end);
}

static void test_mismatch_error() {
    printf("Testing mismatch error handling...\n");
    
    const int32_t vocab_size = 200000;
    const int32_t hidden_size = 1024;
    const int32_t n_tokens = 5;
    const int32_t n_audio_frames = 5;
    
    std::vector<float> token_embd(vocab_size * hidden_size, 0.0f);
    
    int32_t input_ids[] = {151669, 151676, 151676, 151676, 151670};
    
    std::vector<float> audio_features(n_audio_frames * hidden_size, 0.0f);
    
    audio_injection_context ctx;
    ctx.token_embd = token_embd.data();
    ctx.vocab_size = vocab_size;
    ctx.hidden_size = hidden_size;
    ctx.tokens.audio_pad_token_id = 151676;
    
    injection_result result = inject_audio(
        input_ids, n_tokens,
        audio_features.data(), n_audio_frames,
        ctx);
    
    assert(!result.success);
    assert(!result.error_msg.empty());
    
    printf("  PASS: Correctly detected mismatch (3 pads vs 5 frames)\n");
    printf("    Error: %s\n", result.error_msg.c_str());
}

static void test_no_audio() {
    printf("Testing no audio case...\n");
    
    const int32_t vocab_size = 200000;
    const int32_t hidden_size = 1024;
    const int32_t n_tokens = 3;
    
    std::vector<float> token_embd(vocab_size * hidden_size);
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            token_embd[i * hidden_size + j] = static_cast<float>(i % 1000);
        }
    }
    
    int32_t input_ids[] = {100, 200, 300};
    
    audio_injection_context ctx;
    ctx.token_embd = token_embd.data();
    ctx.vocab_size = vocab_size;
    ctx.hidden_size = hidden_size;
    
    injection_result result = inject_audio(
        input_ids, n_tokens,
        nullptr, 0,
        ctx);
    
    assert(result.success);
    assert(result.seq_len == n_tokens);
    
    assert(float_eq(result.embeddings[0 * hidden_size], 100.0f, 0.01f));
    assert(float_eq(result.embeddings[1 * hidden_size], 200.0f, 0.01f));
    assert(float_eq(result.embeddings[2 * hidden_size], 300.0f, 0.01f));
    
    printf("  PASS: No audio case works correctly\n");
}

static void test_count_and_find() {
    printf("Testing count_audio_pad_tokens and find_audio_start_position...\n");
    
    int32_t input_ids[] = {100, 151669, 151676, 151676, 151676, 151670, 200};
    int32_t n_tokens = 7;
    int32_t audio_pad_token_id = 151676;
    
    int32_t count = count_audio_pad_tokens(input_ids, n_tokens, audio_pad_token_id);
    assert(count == 3);
    
    int32_t start_pos = find_audio_start_position(input_ids, n_tokens, audio_pad_token_id);
    assert(start_pos == 2);
    
    printf("  PASS: count=%d, start_pos=%d\n", count, start_pos);
}

static void test_validate_audio_injection() {
    printf("Testing validate_audio_injection...\n");
    
    int32_t input_ids[] = {151669, 151676, 151676, 151676, 151670};
    int32_t n_tokens = 5;
    int32_t audio_pad_token_id = 151676;
    std::string error_msg;
    
    bool valid = validate_audio_injection(input_ids, n_tokens, 3, audio_pad_token_id, error_msg);
    assert(valid);
    
    valid = validate_audio_injection(input_ids, n_tokens, 5, audio_pad_token_id, error_msg);
    assert(!valid);
    assert(!error_msg.empty());
    
    printf("  PASS: Validation works correctly\n");
}

int main() {
    printf("=== Audio Injection Tests ===\n\n");
    
    test_find_audio_positions();
    test_embed_tokens();
    test_inject_audio_embeddings();
    test_inject_audio_full();
    test_mismatch_error();
    test_no_audio();
    test_count_and_find();
    test_validate_audio_injection();
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
