cmake_minimum_required(VERSION 2.8)
project( mono-vo )


if(WIN32)
	find_path(OpenCV_DIR OpenCVConfig.cmake PATHS 
			  "C:/OpenCV/opencv-3.1.0/build" # PC Amaury Mines
			  # add your OpenCV path here : 
			  )
	if( ${OpenCV_DIR} STREQUAL "OpenCV_DIR-NOTFOUND" ) 
		message( FATAL_ERROR "FATAL_ERROR : OpenCV is not found, Please insert your OpenCV Path in pl_utils.cmake")
	else ()
		message(STATUS "Your OpenCV_DIR is ${OpenCV_DIR}")
	endif()
	find_package( OpenCV 3.0 REQUIRED )
else()
	find_package( OpenCV 3.0 REQUIRED )
endif()



include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)
add_executable( vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS} )
