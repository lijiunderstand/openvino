// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "intel_gpu/runtime/device.hpp"

#include "ocl_device.hpp"
#include "ocl_kernel.hpp"


namespace cldnn {
namespace ocl {

class ocl_kernel_builder : public kernel_builder{
    public:
        ocl_kernel_builder(const ocl_device &device) : m_device(device) {}

        void build_kernels(const void *src,
            size_t src_bytes,
            KernelFormat src_format,
            const std::string &options,
            std::vector<kernel::ptr> &out) const override {
            auto context = m_device.get_context().get();

            cl_program program_handle;
            cl_int err = CL_INVALID_VALUE;
            switch (src_format) {
            case KernelFormat::SOURCE: {
                const char **strings = reinterpret_cast<const char**>(&src);
                const size_t *lenghts = &src_bytes;
                const cl_uint count = 1;
                program_handle = clCreateProgramWithSource(context, count, strings, lenghts, &err);
                break;
            }
            case KernelFormat::NATIVE_BIN: {
                const unsigned char **binaries = reinterpret_cast<const unsigned char**>(&src);
                const size_t *lenghts = &src_bytes;
                const cl_device_id device_id = m_device.get_device().get();
                const cl_uint count = 1;
                program_handle = clCreateProgramWithBinary(context, count, &device_id, lenghts, binaries, nullptr, &err);
                break;
            }
            default:
                OPENVINO_THROW("[GPU] Trying to build kernel from unexpected format");
                break;
            }
            if (err != CL_SUCCESS) {
                OPENVINO_THROW("[GPU] Failed to create program during kernel build process");
            }
            cl::Program program(program_handle);
            cl_int build_err = CL_SUCCESS;
            try {
                build_err = program.build({m_device.get_device()}, options.c_str());
            } catch (const cl::Error& e) {
                auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
                std::string build_log;
                for (auto &entry : log) {
                    build_log += entry.second;
                }
                OPENVINO_THROW("[GPU] Failed to build program. CL error: ", e.what(),
                               " (", e.err(), ")\nOpenCL build log:\n", build_log);
            }

            if (build_err != CL_SUCCESS) {
                GPU_DEBUG_INFO << "-------- Kernel build error" << std::endl;
                auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
                std::string build_log;
                for (auto &e : log) {
                    GPU_DEBUG_INFO << e.second;
                    build_log += e.second;
                }
                GPU_DEBUG_INFO << "-------- End of Kernel build error" << std::endl;
                OPENVINO_THROW("[GPU] Failed to build program. OpenCL build log:\n", build_log);
            }
            cl::vector<cl::Kernel> kernels;
            if (program.createKernels(&kernels) != CL_SUCCESS) {
                OPENVINO_THROW("[GPU] Failed to create kernels");
            }
            for (auto& k : kernels) {
                const auto &entry_point = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                out.push_back(std::make_shared<ocl::ocl_kernel>(ocl::ocl_kernel_type(k, m_device.get_usm_helper()), entry_point));
            }
    }

    private:
        const ocl_device &m_device;
};
}  // namespace ocl
}  // namespace cldnn

