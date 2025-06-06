function(export_resnet_model DATASET_NAME OUTPUT_FILENAME NUM_CLASSES PREFIX)
    message(STATUS "Exporting ResNet-18 model to train on ${DATASET_NAME} dataset...")

    run_python_script_with_auto_install(
            SCRIPT "${PY_SCRIPT_PATH}/resnet18.py" #TODO: usare modello pytorch
            ARGS ${OUTPUT_FILENAME} ${NUM_CLASSES}
            RESULT_VARIABLE export_success
            ERROR_VARIABLE export_error
    )

    if(export_success)
        message(STATUS "ResNet-18 for ${DATASET_NAME} successfully exported.")
        add_compile_definitions(${PREFIX}_FILENAME="${OUTPUT_FILENAME}.pt")
        add_compile_definitions(${PREFIX}_NUM_CLASSES="${NUM_CLASSES}")
    else()
        message(FATAL_ERROR "Error exporting ResNet-18 model for ${DATASET_NAME}:\n${export_error}")
    endif()
endfunction()
