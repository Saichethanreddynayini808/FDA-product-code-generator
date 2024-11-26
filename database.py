import sqlite3

from flask import Flask, render_template
import sqlite3

app = Flask(__name__)


# Function to fetch all rows from the database
def fetch_all_fda_codes():
    conn = sqlite3.connect('fda_codes.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM fda_codes')
    rows = cursor.fetchall()
    conn.close()
    return rows


@app.route('/view_fda_codes')
def view_fda_codes():
    # Fetch data from the database
    data = fetch_all_fda_codes()

    # Pass the data to the HTML template
    return render_template('index2.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)

