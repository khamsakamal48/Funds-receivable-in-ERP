# Funds receivable in ERP

```postgresql
CREATE DATABASE erp_data;

\c erp_data

CREATE SCHEMA funds_received;

SET search_path TO funds_received;

CREATE TABLE project_directory (
    project_id VARCHAR PRIMARY KEY,
    project_name VARCHAR,
    category VARCHAR
);
```