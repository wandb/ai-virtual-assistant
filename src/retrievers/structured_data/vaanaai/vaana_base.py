# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from vanna.milvus import Milvus_VectorStore
from pymilvus import MilvusClient
import logging
import sqlparse
import re
import os
import pandas as pd
from jinja2 import Template
from typing import Union
from src.retrievers.structured_data.vaanaai.utils import NVIDIAEmbeddingsWrapper
from src.retrievers.structured_data.vaanaai.vaana_llm import NvidiaLLM
from src.common.utils import get_embedding_model, get_config, get_prompts

logger = logging.getLogger(__name__)
prompts = get_prompts()

class VannaWrapper(Milvus_VectorStore, NvidiaLLM):
    def __init__(self, config=None):
        logger.info("Initializing MyVanna with NvidiaLLM and Milvus_VectorStore")
        document_embedder = get_embedding_model()
        emb_function = NVIDIAEmbeddingsWrapper(document_embedder)
        settings = get_config()
        if settings.vector_store.name == "milvus":
            milvus_db_url = settings.vector_store.url
        milvus_client = MilvusClient(uri=milvus_db_url)
        Milvus_VectorStore.__init__(self, config={"embedding_function": emb_function, "milvus_client": milvus_client})
        NvidiaLLM.__init__(self, config=config)

    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        """
        Connect to postgres using the psycopg2 connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_postgres(
            host="myhost",
            dbname="mydatabase",
            user="myuser",
            password="mypassword",
            port=5432
        )
        ```
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
        """

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise Exception(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[postgres]"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise Exception("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("POSTGRES_DB")

        if not dbname:
            raise Exception("Please set your postgres database")

        if not user:
            user = os.getenv("POSTGRES_USER")

        if not user:
            raise Exception("Please set your postgres user")

        if not password:
            password = os.getenv("POSTGRES_PASSWORD")

        if not password:
            raise Exception("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise Exception("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs
            )
        except psycopg2.Error as e:
            raise Exception(e)

        def connect_to_db():
            conn = psycopg2.connect(host=host, dbname=dbname,
                        user=user, password=password, port=port, **kwargs)
            # Set the connection to read-only mode
            conn.set_session(readonly=True)
            return conn

        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = connect_to_db()  # Initial connection attempt
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                # Attempt to reconnect and retry the operation
                if conn:
                    conn.close()  # Ensure any existing connection is closed
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise Exception(e)

            except Exception as e:
                        conn.rollback()
                        raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres
    
    def is_sql_valid(self, sql: str, customer_id: str) -> bool:
        """
        Checks if the SQL query is valid. This function validates that the query:
        - Is a SELECT statement.
        - Contains a WHERE clause filtering on the given customer_id.
    
        Args:
            sql (str): The SQL query to check.
            customer_id (str): The customer_id to validate in the WHERE clause.
    
        Returns:
            bool: True if the SQL query is valid and contains a matching customer_id filter, False otherwise.
        """
        parsed = sqlparse.parse(sql)
        sql_validation = False
        customer_id_validation = False
    
        for statement in parsed:
            # Check if it's a SELECT statement
            if statement.get_type() == 'SELECT':
                logger.info(f"The SQL Statement {str(statement)} is of type SELECT and is safe")
                sql_validation = True
    
                # Convert the statement to a string for regex-based parsing
                statement_str = str(statement)

                # Check if there's a WHERE clause with the specified customer_id filter
                where_clause_match = re.search(r"WHERE\s+.*customer_id\s*=\s*['\"]?" + re.escape(str(customer_id)) + r"['\"]?", statement_str, re.IGNORECASE)
                # Find all occurrences of customer_id conditions in the statement
                customer_id_matches = re.findall(r"customer_id\s*=\s*['\"]?(\d+)['\"]?", statement_str, re.IGNORECASE)
                logger.info(f"WHERE clause: {where_clause_match}, is matched with customer_id: {customer_id}")
                logger.info(f"Number of Customer IDs: {customer_id_matches}")
                if where_clause_match and len(customer_id_matches) == 1:
                    customer_id_validation = True
                logger.info(f"customer_id_validation: {customer_id_validation}")
    
        # Return True only if both the SQL validation and customer_id validation passed
        return sql_validation and customer_id_validation
    

    def _get_ddl_data(self) -> list[dict]:

        tables: dict[str, dict] = {}

        res_sql_query = """
            WITH relevant_tables AS (
                SELECT
                    table_name,
                    table_schema
                FROM
                    information_schema.tables
                WHERE
                    table_schema = 'public'
            )
            SELECT
                relevant_tables.table_name AS table_name,
                c.column_name AS col_name,
                UPPER(data_type) AS col_type
            FROM
                information_schema.columns c
            INNER JOIN
                relevant_tables
            ON
                relevant_tables.table_schema = c.table_schema
                AND relevant_tables.table_name = c.table_name
            ORDER BY
                col_name ASC
        """
        res = self.run_sql(res_sql_query)
        for _, row in res.iterrows():
            table_name = row["table_name"]
            col_name = row["col_name"]
            column = {
                "name": col_name,
                "type": row["col_type"],
            }
            if table_name in tables:
                tables[table_name]["columns"].append(column)
            else:
                tables[table_name] = {
                    "name": table_name,
                    "columns": [column],
                }

        return list(tables.values())

    def do_training(self, method: str = "ddl"):
        
        if self.get_training_data().empty:
            logger.info(f"Training metnod: {method}")
            logger.info(f"No Training data found, training with {method}")
            if method == "ddl":
                # Define the Jinja template
                template_str = """
                {% for table in tables %}
                CREATE TABLE {{ table.name }} (
                    {% for column in table.columns %}
                    "{{ column.name }}" {{ column.type }}{% if not loop.last or table.primary_key %},{% endif %}
                    {% endfor %}
                    {% if table.primary_key %}
                    PRIMARY KEY ({{ table.primary_key }})
                    {% endif %}
                    {% if table.foreign_keys %},
                    {% for foreign_key in table.foreign_keys %}
                    FOREIGN KEY ({{ foreign_key.from }}) REFERENCES {{ foreign_key.table }} ({{ foreign_key.to }}){% if not loop.last %},{% endif %}
                    {% endfor %}
                    {% endif %}
                );
                {% if not loop.last %}
                {% endif %}
                {% endfor %}
                """
                ddl_data = self._get_ddl_data()
                logger.info(f"TRAINING DATA: \n {ddl_data}")
                template = Template(template_str)
                ddl = template.render(tables=ddl_data)
                self.train(ddl=ddl)
            elif method == "schema":
                static_ddl_schema = prompts.get("static_db_schema", "")
                if static_ddl_schema:
                    logger.info(f"TRAINING DATA: \n {static_ddl_schema}")
                    self.train(ddl=static_ddl_schema)
                else:
                    logger.info(f"Skipping training as no static_db_schema found in prompts.yaml")
        else:
            logger.info(f"Training not needed, data already found:\n {self.get_training_data()}")

    def ask_query(self, question: str, user_id: str) -> Union[pd.DataFrame, str]:
        """
        Generates and validates an SQL query based on a given question and customer ID, 
        then executes the query if valid.

        Parameters:
            question (str): The input question to be converted into an SQL query.
            user_id (str): The ID of the customer, used for query validation.

        Returns:
            Union[pd.DataFrame, str]: The result of the SQL query as a DataFrame if successful,
            or a string indicating the query is invalid or if an error occurred.
        """

        try:
            query = question + f", for user_id: {user_id}"
            logger.info(f"input query with user_id: {query}")
            
            sql = self.generate_sql(question=query, allow_llm_to_see_data=True)
            logger.info(f"generated sql: {sql}")
            
            sql_valid = self.is_sql_valid(sql=sql, customer_id=user_id)
            logger.info(f"Is SQL valid: {sql_valid}")
            
            if sql_valid:
                output_df = self.run_sql(sql=sql)
                logger.info(f"output df: {output_df}")
                return output_df
            else:
                logger.warning("SQL is not valid")
                return "not valid sql"
        except Exception as e:
            logger.error(f"Error occurred in ask_query: {e}")
            raise  # Re-raise the exception after logging
