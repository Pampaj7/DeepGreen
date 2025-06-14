# The download_and_convert_dataset() function runs a Python script, specified
# by the PY_SCRIPT_NAME argument, located at ../dataloader directory. This script
# has to download and convert a dataset to PNGs images.
# The OUTPUT_DIR argument specifies where dataset is saved at ../data directory.
#
function(download_and_convert_dataset DATASET_NAME OUTPUT_DIR PY_SCRIPT_NAME)
    set(DATASET_ROOT "${CMAKE_CURRENT_LIST_DIR}/../data")

    if (WIN32)
        message(WARNING "For Windows users, the following script may fail due to using too long directory paths during conversion.\nIn this case, please, enable long paths in Windows.")
    endif ()
    message(STATUS "Downloading and converting ${DATASET_NAME} dataset...")
    run_python_script_with_auto_install(
            SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../dataloader/${PY_SCRIPT_NAME}.py"
            ARGS "${DATASET_ROOT}/${OUTPUT_DIR}" #${DATASET_ROOT}
            RESULT_VARIABLE success
            ERROR_VARIABLE error
    )
    if(success)
        message(STATUS "${DATASET_NAME} successfully downloaded and converted.")
    else()
        message(FATAL_ERROR "Error downloading or converting ${DATASET_NAME}:\n${error}")
    endif()
endfunction()