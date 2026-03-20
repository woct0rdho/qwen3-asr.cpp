#include "gguf_loader.h"

#include <cstdio>
#include <vector>
#include <ggml.h>
#include <ggml-backend.h>

int main() {
    // Load model
    qwen3_asr::audio_encoder_model model;
    qwen3_asr::GGUFLoader loader;
    if (!loader.load("models/qwen3-asr-0.6b-f32.gguf", model)) {
        printf("Failed to load model: %s\n", loader.get_error().c_str());
        return 1;
    }
    
    printf("Model loaded\n");
    
    // Check conv1 weight
    struct ggml_tensor* conv1_w = model.conv2d1_w;
    printf("\nConv1 weight:\n");
    printf("  Shape: [%lld, %lld, %lld, %lld]\n", 
           (long long)conv1_w->ne[0], (long long)conv1_w->ne[1], 
           (long long)conv1_w->ne[2], (long long)conv1_w->ne[3]);
    printf("  Type: %d\n", conv1_w->type);
    
    // Read kernel values
    // GGML kernel is [KW, KH, IC, OC] = [3, 3, 1, 480]
    // Element [kw, kh, ic, oc] is at offset kw + kh*3 + ic*9 + oc*9
    
    std::vector<float> kernel_data(3 * 3 * 1 * 480);
    ggml_backend_tensor_get(conv1_w, kernel_data.data(), 0, kernel_data.size() * sizeof(float));
    
    printf("\nKernel filter 0 (oc=0), indexed as [kw, kh]:\n");
    for (int kh = 0; kh < 3; kh++) {
        printf("  ");
        for (int kw = 0; kw < 3; kw++) {
            int idx = kw + kh * 3 + 0 * 9 + 0 * 9;
            printf("%9.6f ", kernel_data[idx]);
        }
        printf("\n");
    }
    
    // Expected (from PyTorch kernel[0,0,:,:] in C order, kw varies fastest):
    // [[-0.003433 -0.037109 -0.117676]
    //  [-0.039062 -0.018311  0.367188]
    //  [ 0.005127 -0.037842 -0.218750]]
    
    printf("\nExpected (PyTorch kernel[0,0,:,:], indexed as [kw, kh]):\n");
    printf("  -0.003433 -0.037109 -0.117676\n");
    printf("  -0.039062 -0.018311  0.367188\n");
    printf("   0.005127 -0.037842 -0.218750\n");
    
    // Check bias
    struct ggml_tensor* conv1_b = model.conv2d1_b;
    std::vector<float> bias_data(480);
    ggml_backend_tensor_get(conv1_b, bias_data.data(), 0, bias_data.size() * sizeof(float));
    printf("\nConv1 bias[0]: %.6f (expected: -0.062256)\n", bias_data[0]);
    
    qwen3_asr::free_model(model);
    return 0;
}
