# The install_codecarbon() function runs a Python script that imports codecarbon.
# If the import doesn't return any failure, codecarbon is already installed in the Python env.
# Otherwise, codecarbon will be installed through pip.
#
function(install_codecarbon)
    run_python_script_with_auto_install(
            SCRIPT "${PY_SCRIPT_PATH}/tracker/import_codecarbon.py"
            RESULT_VARIABLE import_success
            ERROR_VARIABLE import_error
    )
    if(import_success)
        message(STATUS "Codecarbon successfully imported.")
    else()
        message(FATAL_ERROR "Error importing codecarbon:\n${import_error}")
    endif()
endfunction()