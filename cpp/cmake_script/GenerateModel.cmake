# The export_model_for_dataset() function runs a Python script,
# i.e. PY_SCRIPT_FILENAME.py, to export the model MODEL_NAME
# as PyTorch file (.pt).
# The model is fitted for the given dataset DATASET_NAME, i.e
# the output of last layer is changed in accord to NUM_CLASSES.
#
function(export_model_for_dataset MODEL_NAME DATASET_NAME NUM_CLASSES)
    message(STATUS "Exporting ${MODEL_NAME} model to train on ${DATASET_NAME} dataset...")

    # From MODEL_NAME get the name of the Python script file
    # converting to lowercase and removing non-alphanumerics
    string(REGEX REPLACE "[^A-Za-z0-9]" "" MODEL_FILENAME "${MODEL_NAME}")
    string(TOLOWER "${MODEL_FILENAME}" MODEL_FILENAME)

    # Set the output filename as model_dataset names lowercases
    # and without non-alphanumerics characters
    string(REGEX REPLACE "[^A-Za-z0-9]" "" DATASET_FILENAME "${DATASET_NAME}")
    string(TOLOWER "${DATASET_FILENAME}" DATASET_FILENAME)
    set(OUTPUT_FILENAME "${MODEL_FILENAME}_${DATASET_FILENAME}")

    run_python_script_with_auto_install(
            SCRIPT "${PY_SCRIPT_PATH}/${MODEL_FILENAME}.py"
            ARGS ${OUTPUT_FILENAME} ${NUM_CLASSES}
            RESULT_VARIABLE export_success
            ERROR_VARIABLE export_error
    )

    if(export_success)
        message(STATUS "${MODEL_NAME} for ${DATASET_NAME} successfully exported.")

        # Convert OUTPUT_FILENAME to uppercase
        string(TOUPPER "${OUTPUT_FILENAME}" FILENAME_PREFIX)
        add_compile_definitions(${FILENAME_PREFIX}_FILENAME="${OUTPUT_FILENAME}.pt")
    else()
        message(FATAL_ERROR "Error exporting ${MODEL_NAME} model for ${DATASET_NAME}:\n${export_error}")
    endif()
endfunction()