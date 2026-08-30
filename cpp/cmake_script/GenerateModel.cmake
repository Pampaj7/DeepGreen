# Bring the shared TorchScript module into the build tree.
#
# This function used to *export* the model: it ran cpp/py_script/models/<arch>.py
# to produce a .pt in CMAKE_BINARY_DIR. That is a second export, from a second
# definition, with no seed -- so C++ and Rust each trained a network built from
# a different random draw, while the manuscript described them, together with
# Python/PyTorch, as training a byte-identical module and the collapse analysis
# rested on their starting parameters being equal. Compared tensor by tensor the
# two files disagreed on every layer.
#
# One export now: scripts/export_torchscript_models.py, seeded, with
# models/MANIFEST.json recording the parameter count and a hash of the weights.
# This copies that artefact in rather than making another one.
#
function(export_model_for_dataset MODEL_NAME DATASET_NAME NUM_CLASSES)
    # From MODEL_NAME get the module's filename stem, lowercased and stripped
    # of non-alphanumerics, as the shared export names it.
    string(REGEX REPLACE "[^A-Za-z0-9]" "" MODEL_FILENAME "${MODEL_NAME}")
    string(TOLOWER "${MODEL_FILENAME}" MODEL_FILENAME)
    string(REGEX REPLACE "[^A-Za-z0-9]" "" DATASET_FILENAME "${DATASET_NAME}")
    string(TOLOWER "${DATASET_FILENAME}" DATASET_FILENAME)
    set(OUTPUT_FILENAME "${MODEL_FILENAME}_${DATASET_FILENAME}")

    # DEEPGREEN_MODELS if set, so a replicator can point the build at an
    # export it produced itself; the repository's models/ otherwise.
    if(DEFINED ENV{DEEPGREEN_MODELS})
        set(SHARED_MODELS "$ENV{DEEPGREEN_MODELS}")
    else()
        set(SHARED_MODELS "${CMAKE_CURRENT_SOURCE_DIR}/../models")
    endif()
    set(SOURCE_MODULE "${SHARED_MODELS}/${OUTPUT_FILENAME}.pt")

    if(NOT EXISTS "${SOURCE_MODULE}")
        message(FATAL_ERROR
                "Shared module ${SOURCE_MODULE} is missing.\n"
                "Run scripts/export_torchscript_models.py first: the stacks that "
                "share a backend must share the module, not each build its own.")
    endif()

    message(STATUS "Using shared module ${SOURCE_MODULE}")
    file(COPY "${SOURCE_MODULE}" DESTINATION "${CMAKE_BINARY_DIR}")

    string(TOUPPER "${OUTPUT_FILENAME}" FILENAME_PREFIX)
    add_compile_definitions(${FILENAME_PREFIX}_FILENAME="${OUTPUT_FILENAME}.pt")
endfunction()
