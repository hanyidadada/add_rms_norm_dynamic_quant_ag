add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        add_rms_norm_dynamic_quant_ag_def.cpp
    )
endif()

add_ops_compile_options(
        OP_NAME AddRmsNormDynamicQuantAG
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
                -Werror
                -DHCCL_COMM
)


if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE add_rms_norm_dynamic_quant_ag ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()