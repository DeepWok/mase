configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  MAIN_CONFIG
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  )

set(MASE_TEST_DEPENDS
  FileCheck count not
  mase-opt
  )

add_lit_testsuite(check-mase "Running the Mase regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MASE_TEST_DEPENDS}
  )
set_target_properties(check-mase PROPERTIES FOLDER "Tests")

add_lit_testsuites(MASE ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MASE_TEST_DEPS}
)

