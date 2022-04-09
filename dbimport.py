# -*- coding: utf-8 -*-

import sqlite3


def getrow(row_id: int, dbfile: str) -> list:
    """Gets one row from the training database
    
        Parameters
        ----------
        row_id : int
            The ID of the desired row
        dbfile : str
            The name of the database file
            
        Returns
        -------
        list
            A list of rows returned from the query as tuples
            
    """
    
    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    cur.execute("SELECT * FROM evaluations WHERE id=:id;", {"id": row_id})
    data = cur.fetchall()
    con.close()
    return data
    
def countdata(dbfile: str) -> int:
    """Counts the number of rows in the database
    
        Parameters
        ----------
        dbfile : str
            The name of the database file
            
        Returns
        -------
        int
            The number of rows in the database
            
    """
    
    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    cur.execute("SELECT COUNT(id) FROM evaluations;")
    data = cur.fetchall()
    con.close()
    
    # fetchall returns the count as an int inside a tuple inside a list
    # this extracts and returns just the int value
    
    return data[0][0]
    
def getrowbatch(row_id: int, batch_size: int, dbfile: str) -> list:
    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    cur.execute("SELECT * FROM evaluations WHERE id>:id AND id<=:count", {"id": row_id, "count": batch_size + row_id})
    data = cur.fetchall()
    con.close()
    
    return data