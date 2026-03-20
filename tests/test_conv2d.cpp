#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

int main() {
    // Create a simple test case
    // Input: [1, 1, 4, 4] (N=1, IC=1, IH=4, IW=4)
    // Kernel: [1, 1, 3, 3] (OC=1, IC=1, KH=3, KW=3)
    // Stride: 1, Padding: 1
    // Output: [1, 1, 4, 4]
    
    const int N = 1, IC = 1, IH = 4, IW = 4;
    const int OC = 1, KH = 3, KW = 3;
    const int stride = 1, padding = 1;
    
    // Input data (simple pattern)
    float input_data[N * IC * IH * IW];
    for (int i = 0; i < IH * IW; i++) {
        input_data[i] = (float)(i + 1);  // 1, 2, 3, ..., 16
    }
    
    // Kernel data (simple pattern)
    // PyTorch kernel [OC, IC, KH, KW] = [1, 1, 3, 3]
    float kernel_pt[OC * IC * KH * KW] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    
    // GGML expects [KW, KH, IC, OC]
    // So we need to transpose the kernel
    float kernel_ggml[KW * KH * IC * OC];
    for (int oc = 0; oc < OC; oc++) {
        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    // GGML index: kw + kh*KW + ic*KW*KH + oc*KW*KH*IC
                    // PyTorch index: oc*IC*KH*KW + ic*KH*KW + kh*KW + kw
                    int ggml_idx = kw + kh*KW + ic*KW*KH + oc*KW*KH*IC;
                    int pt_idx = oc*IC*KH*KW + ic*KH*KW + kh*KW + kw;
                    kernel_ggml[ggml_idx] = kernel_pt[pt_idx];
                }
            }
        }
    }
    
    printf("Input (IH x IW = 4 x 4):\n");
    for (int h = 0; h < IH; h++) {
        for (int w = 0; w < IW; w++) {
            printf("%5.1f ", input_data[h * IW + w]);
        }
        printf("\n");
    }
    
    printf("\nKernel (PyTorch, KH x KW = 3 x 3):\n");
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            printf("%5.1f ", kernel_pt[kh * KW + kw]);
        }
        printf("\n");
    }
    
    printf("\nKernel (GGML, KW x KH = 3 x 3):\n");
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            printf("%5.1f ", kernel_ggml[kw + kh * KW]);
        }
        printf("\n");
    }
    
    // Expected output (manual calculation for position (0,0)):
    // With padding=1, the input region for output (0,0) is:
    // (-1,-1) (-1,0) (-1,1)   ->  0  0  0
    // (0,-1)  (0,0)  (0,1)    ->  0  1  2
    // (1,-1)  (1,0)  (1,1)    ->  0  5  6
    // Convolution: 0*1 + 0*2 + 0*3 + 0*4 + 1*5 + 2*6 + 0*7 + 5*8 + 6*9 = 5 + 12 + 40 + 54 = 111
    // Wait, that's wrong. Let me recalculate...
    // Kernel is [kh, kw], input region is [ih, iw]
    // For output (oh=0, ow=0) with stride=1, padding=1:
    // ih = oh*stride - padding + kh = 0 - 1 + kh = kh - 1
    // iw = ow*stride - padding + kw = 0 - 1 + kw = kw - 1
    // So:
    // kh=0, kw=0: ih=-1, iw=-1 -> 0 * kernel[0,0]=1 = 0
    // kh=0, kw=1: ih=-1, iw=0  -> 0 * kernel[0,1]=2 = 0
    // kh=0, kw=2: ih=-1, iw=1  -> 0 * kernel[0,2]=3 = 0
    // kh=1, kw=0: ih=0, iw=-1  -> 0 * kernel[1,0]=4 = 0
    // kh=1, kw=1: ih=0, iw=0   -> 1 * kernel[1,1]=5 = 5
    // kh=1, kw=2: ih=0, iw=1   -> 2 * kernel[1,2]=6 = 12
    // kh=2, kw=0: ih=1, iw=-1  -> 0 * kernel[2,0]=7 = 0
    // kh=2, kw=1: ih=1, iw=0   -> 5 * kernel[2,1]=8 = 40
    // kh=2, kw=2: ih=1, iw=1   -> 6 * kernel[2,2]=9 = 54
    // Total: 5 + 12 + 40 + 54 = 111
    
    printf("\nExpected output[0,0,0,0] = 111\n");
    
    // Initialize GGML
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    
    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    // Create tensors
    // Input: GGML expects [IW, IH, IC, N]
    struct ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, IC, N);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    
    // Kernel: GGML expects [KW, KH, IC, OC]
    struct ggml_tensor * kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, IC, OC);
    ggml_set_name(kernel, "kernel");
    ggml_set_input(kernel);
    
    // Conv2d
    struct ggml_tensor * output = ggml_conv_2d(ctx, kernel, input, stride, stride, padding, padding, 1, 1);
    ggml_set_name(output, "output");
    ggml_set_output(output);
    
    // Build graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    
    // Allocate
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    // Set input data
    // GGML input is [IW, IH, IC, N], element [w, h, c, n] is at offset w + h*IW + c*IW*IH + n*IW*IH*IC
    // PyTorch input is [N, IC, IH, IW], element [n, c, h, w] is at offset w + h*IW + c*IH*IW + n*IC*IH*IW
    // So GGML [w, h, 0, 0] = PyTorch [0, 0, h, w] = input_data[h * IW + w]
    // The memory layout is the same!
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(kernel, kernel_ggml, 0, sizeof(kernel_ggml));
    
    // Compute
    ggml_backend_graph_compute(backend, graph);
    
    // Get output
    int OH = (IH + 2*padding - KH) / stride + 1;
    int OW_out = (IW + 2*padding - KW) / stride + 1;
    printf("\nOutput shape: [%d, %d, %d, %d]\n", N, OC, OH, OW_out);
    
    std::vector<float> output_data(N * OC * OH * OW_out);
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));
    
    printf("Output (OH x OW = %d x %d):\n", OH, OW_out);
    for (int h = 0; h < OH; h++) {
        for (int w = 0; w < OW_out; w++) {
            printf("%6.1f ", output_data[w + h * OW_out]);
        }
        printf("\n");
    }
    
    printf("\nGGML output[0,0,0,0] = %.1f\n", output_data[0]);
    
    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    
    return 0;
}
