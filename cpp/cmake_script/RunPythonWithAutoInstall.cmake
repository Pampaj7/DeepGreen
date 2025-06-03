# The run_python_script_with_auto_install() function can be used to
# run python scripts, even with arguments. If the script needs some
# dependencies, this function can install them via pip command, up
# to a maximum of 10 dependencies before failing.
#
#    run_python_script_with_auto_install(
#         SCRIPT <"path/to/script.py">
#         [ARGS <arg1> [<arg2> ... ]]
#         [RESULT_VARIABLE <result_var_name>]
#         [ERROR_VARIABLE <error_var_name>]
#    )
#
function(run_python_script_with_auto_install)
    cmake_parse_arguments(RUNPY "" "SCRIPT;RESULT_VARIABLE;ERROR_VARIABLE" "ARGS" ${ARGN})

    if(NOT RUNPY_SCRIPT)
        message(FATAL_ERROR "You have to specify the python script through: SCRIPT \"path/to/script.py\".")
    endif()

    set(MAX_ATTEMPTS 10)
    set(EXEC_OK FALSE)

    foreach(i RANGE ${MAX_ATTEMPTS})
        execute_process(
                COMMAND ${Python3_EXECUTABLE} ${RUNPY_SCRIPT} ${RUNPY_ARGS}
                RESULT_VARIABLE result
                ERROR_VARIABLE error_output
                ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(result EQUAL 0)
            set(EXEC_OK TRUE)
            break()
        endif()

        # Controlla se c'è un ModuleNotFoundError
        string(REGEX MATCH "ModuleNotFoundError: No module named '([^']+)'" match "${error_output}")

        if(match)
            string(REGEX REPLACE ".*ModuleNotFoundError: No module named '([^']+)'.*" "\\1" missing_module "${error_output}")
            message(WARNING "Missing module: '${missing_module}'. Try to install it...")

            execute_process(
                    COMMAND ${Python3_EXECUTABLE} -m pip install ${missing_module}
                    RESULT_VARIABLE pip_result
                    ERROR_VARIABLE pip_error
            )

            if(pip_result EQUAL 0)
                message(STATUS "Module '${missing_module}' successfully installed.")
            else ()
                set(EXEC_ERROR "Error installing '${missing_module}':\n${pip_error}")
            endif()
        else()
            set(EXEC_ERROR "Error in script not related to missing modules:\n${error_output}")
        endif()
    endforeach()

    if(RUNPY_RESULT_VARIABLE)
        set(${RUNPY_RESULT_VARIABLE} ${EXEC_OK} PARENT_SCOPE)
    endif()

    if(RUNPY_ERROR_VARIABLE)
        if(EXEC_OK)
            set(${RUNPY_ERROR_VARIABLE} "" PARENT_SCOPE)
        else ()
            set(${RUNPY_ERROR_VARIABLE} ${EXEC_ERROR} PARENT_SCOPE)
        endif ()
    endif ()

    if(NOT EXEC_OK AND NOT RUNPY_RESULT_VARIABLE AND NOT RUNPY_ERROR_VARIABLE)
        message(FATAL_ERROR ${EXEC_ERROR})
    endif()
endfunction()