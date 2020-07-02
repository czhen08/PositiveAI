import pyodbc
server = 'positiveai.database.windows.net'
database = 'positiveAIdb'
username = 'positiveaiadmin'
password = 'Zcdukic001'
driver= '{ODBC Driver 17 for SQL Server}'
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

# table_name = 'Person'

# cursor.execute("SELECT * FROM "+table_name)
# rows = cursor.fetchall()
# for row in rows:
#     print(row.PersonId,row.FirstName, row.LastName, row.DateOfBirth)
#     print(type(row.PersonId))
#     print(type(row.FirstName))