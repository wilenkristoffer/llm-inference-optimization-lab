/* Return 0 == ROCPROFILER_REGISTER_SUCCESS, so the caller carries on. */
int rocprofiler_register_library_api_table(const char *lib_name,
                                           const void *import_funcs,
                                           unsigned int lib_version,
                                           void **tables,
                                           unsigned long num_tables)
{
    (void)lib_name; (void)import_funcs; (void)lib_version;
    (void)tables;   (void)num_tables;
    return 0;
}
