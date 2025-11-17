import os
import json
import pandas as pd
import duckdb
import time

class Utility:
    """Utility class to provide common helper methods."""

    @staticmethod
    def save_table(path, table_name, dataframe, file_format='csv'):
        """
        Save a Pandas DataFrame to a file in the specified format.
        
        Args:
            path (str): Directory path where file will be saved.
            table_name (str): File name (with or without extension).
            dataframe (pd.DataFrame): The DataFrame to save.
            file_format (str): Format to save the file ('csv', 'json', 'parquet').
        """
        # Ensure the filename has the correct extension
        extension_map = {
            "csv": ".csv",
            "json": ".json",
            "parquet": ".parquet"
        }
        if file_format not in extension_map:
            raise ValueError("Unsupported file format. Use 'csv', 'json', or 'parquet'.")
    
        if not table_name.endswith(extension_map[file_format]):
            table_name += extension_map[file_format]
    
        save_path = os.path.join(path, table_name)
    
        try:
            if file_format == "csv":
                dataframe.to_csv(save_path, index=False)
            elif file_format == "json":
                dataframe.to_json(save_path, orient="records", lines=True)
            elif file_format == "parquet":
                dataframe.to_parquet(save_path, index=False)
    
            print(f"File saved successfully to {save_path}.")
        except Exception as e:
            print(f"Error saving file: {e}")
            
    # def save_table(path, table_name, dataframe, file_format ='csv'):
    #     """
    #     Save a Pandas DataFrame to a file in the specified format.
        
    #     Args:
    #         dataframe (pd.DataFrame): The DataFrame to save.
    #         file_path (str): Path to save the file.
    #         file_format (str): Format to save the file ('csv' or 'json').
    #     """
    #     save_path = os.path.join(path, table_name)
    #     try:
    #         if file_format == "csv":
    #             dataframe.to_csv(save_path, index=False)
    #         elif file_format == "json":
    #             dataframe.to_json(save_path, orient="records", lines=True)
    #         else:
    #             raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
    #         print(f"File saved successfully to {save_path}.")
    #     except Exception as e:
    #         print(f"Error saving file: {e}")

    @staticmethod
    def log_start_time():
        """
        Log and return the current start time.
        """
        start_time = time.time()
        print(f"Process started at: {start_time}")
        return start_time

    @staticmethod
    def log_end_time(start_time):
        """
        Log and print the elapsed time since the start.
        
        Args:
            start_time (datetime): The time when the process started.
        """
        end_time = time.time()
        elapsed = (end_time - start_time)/60  # elapsed time in minutes 
        #print(f"Process ended at: {end_time}")
        #print(f"Elapsed time: {elapsed}")
        return elapsed

    @staticmethod
    def load_file(file_path, file_format="csv"):
        """
        Load a file into a Pandas DataFrame.
        
        Args:
            file_path (str): Path to the file to load.
            file_format (str): Format of the file ('csv', 'json', 'parquet').
        
        Returns:
            pd.DataFrame: Loaded data as a Pandas DataFrame.
        """
        try:
            if file_format == "csv":
                return pd.read_csv(file_path)
            elif file_format == "json":
                return pd.read_json(file_path, lines=True)
            elif file_format == "parquet":
                return pd.read_parquet(file_path)
            else:
                raise ValueError("Unsupported file format. Use 'csv', 'json', or 'parquet'.")
        except Exception as e:
            print(f"Error loading file: {e}")
        return None
    # def load_file(file_path, file_format="csv"):
    #     """
    #     Load a file into a Pandas DataFrame.
        
    #     Args:
    #         file_path (str): Path to the file to load.
    #         file_format (str): Format of the file ('csv' or 'json').
        
    #     Returns:
    #         pd.DataFrame: Loaded data as a Pandas DataFrame.
    #     """
    #     try:
    #         if file_format == "csv":
    #             return pd.read_csv(file_path)
    #         elif file_format == "json":
    #             return pd.read_json(file_path, lines=True)
    #         else:
    #             raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
    #     except Exception as e:
    #         print(f"Error loading file: {e}")
    #         return None
            
    @staticmethod
    def extract_column_names(file: str) -> list[str]:
        """
        Extract column names from a given file, converting them to lowercase.
        Uses DuckDB for supported formats (CSV, Parquet, JSON, Excel).
        Falls back to Pandas for unsupported formats (HDF, SQL, SAS).
        
        Parameters:
            file (str): Path to the file.
        Returns:
            list[str]: List of lowercase column names.
        """
        ext = os.path.splitext(file)[1].lower()
        con = duckdb.connect()
    
        try:
            # DuckDB supported
            if ext in [".csv", ".tsv"]:
                query = f"SELECT * FROM read_csv_auto('{file}', SAMPLE_SIZE=0, HEADER=TRUE) LIMIT 0"
            elif ext == ".parquet":
                query = f"SELECT * FROM read_parquet('{file}') LIMIT 0"
            elif ext == ".json":
                query = f"SELECT * FROM read_json_auto('{file}') LIMIT 0"
            elif ext in [".xls", ".xlsx"]:
                query = f"SELECT * FROM read_excel('{file}') LIMIT 0"
            else:
                query = None
    
            if query:
                df = con.execute(query).df()
                return [col.lower() for col in df.columns]
    
            #  Fallback to Pandas for unsupported formats of duckdb
            if ext in [".hdf"]:
                df = pd.read_hdf(file, stop=0)
            elif ext in [".sql"]:
                raise ValueError("SQL requires a DB connection string, not a flat file")
            elif ext in [".sas7bdat"]:
                df = pd.read_sas(file, format='sas7bdat', encoding='iso-8859-1', nrows=0)
            else:
                raise ValueError(f"Unsupported file type: {file}")
    
            return [col.lower() for col in df.columns]
    
        finally:
            con.close()
    # def extract_column_names(file: str) -> list[str]:
    #     """
    #     Extract column names from a given file, converting them to lowercase.
    #     Supports: CSV, TSV, JSON, XML, Excel (XLS/XLSX), HDF, SQL, SAS
    #     Parameters:
    #     file (str): Path to the file.
    #     Returns:
    #     list[str]: List of lowercase column names.
    #     Raises:
    #     ValueError: If the file type is unsupported.
    #     """
    #     # Mapping of file extensions to Pandas read functions
    #     read_functions = {
    #         ('.csv', '.tsv'): lambda f: pd.read_csv(f, nrows=0),
    #         ('.json',): pd.read_json,
    #         ('.xml',): pd.read_xml,
    #         ('.xls', '.xlsx'): pd.read_excel,
    #         ('.hdf',): pd.read_hdf,
    #         ('.sql',): pd.read_sql,
    #         ('.sas7bdat',): lambda f: pd.read_sas(f, format='sas7bdat', encoding='iso-8859-1')
    #     }
    
    #     # Identify the correct function based on the file extension
    #     for extensions, read_func in read_functions.items():
    #         if file.endswith(extensions):
    #             df = read_func(file)
    #             return df.columns.str.lower().tolist()
    
    #     # Raise an error for unsupported file types
    #     raise ValueError(f"Unsupported file type: {file}")
       
    @staticmethod
    def ensure_directory_exists(directory_path):
        """
        Ensure that the specified directory exists. Create it if it doesn't exist.
        
        Args:
            directory_path (str): Path to the directory.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")
        
        return True
        
    @staticmethod
    def file_exists(file_path):
        """
        Check if a file exists at the specified path.
        
        Args:
            file_path (str): The path to the file.
            
        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.isfile(file_path)

    @staticmethod
    def create_file_path(base_path, new_dir, file_name):
        return os.path.join(base_path, new_dir, file_name)
        
    @staticmethod    
    def get_file_name(file_path):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        return file_name
        
