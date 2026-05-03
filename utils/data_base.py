import sqlite3 as sql
import pickle
from configuration import Config
from configuration import get_static_vars_dict


class Database:
    def __init__(self):
        self.conn = sql.connect(Config.GENERATED_VALUES.DB_PATH)
        self.c = self.conn.cursor()
        self.create_precipitation_front_table()
        self.create_time_parameters_table()
        self.create_product_plane0_tracking_table()
        self.save_pickled_config_to_db()

    def save_pickled_config_to_db(self):
        dict_config_to_pickle = get_static_vars_dict(Config)
        pickled_instance = pickle.dumps(dict_config_to_pickle)
        self.c.execute('''CREATE TABLE IF NOT EXISTS PickledConfig (pickled_data BLOB)''')
        self.c.execute("INSERT INTO PickledConfig (pickled_data) VALUES (?)", (pickled_instance,))
        self.conn.commit()

    def save_pickled_microstructure(self, microstructure_instance):
        pickled_instance = pickle.dumps(microstructure_instance)
        self.c.execute('''CREATE TABLE IF NOT EXISTS PickledMicrostructure (pickled_data BLOB)''')
        self.c.execute("INSERT INTO PickledMicrostructure (pickled_data) VALUES (?)", (pickled_instance,))
        self.conn.commit()

    def insert_particle_data(self, particle_type, iteration, data):
        query = """CREATE TABLE {}_iter_{} (z int, y int, x int)""".format(particle_type, str(iteration))
        self.c.execute(query)
        query = "INSERT INTO {}_iter_{} VALUES(?, ?, ?);".format(particle_type, str(iteration))
        data = data.transpose()
        data = self.to_tuple(data)
        self.c.executemany(query, data)

    def create_precipitation_front_table(self):
        self.c.execute(f"""CREATE TABLE precip_front_p (sqrt_time int, position int)""")
        if getattr(Config, "ACTIVES_SECONDARY_EXISTENCE", False):
            self.c.execute("""CREATE TABLE precip_front_s (sqrt_time int, position int)""")

    def insert_precipitation_front(self, sqrt_time, position, sign):
        self.c.execute("INSERT INTO precip_front_{} VALUES ({}, {})".format(sign, sqrt_time, position))

    def create_time_parameters_table(self):
        self.c.execute("""CREATE TABLE time_parameters (last_i int, elapsed_time float)""")
        query = """INSERT INTO time_parameters VALUES(?, ?);"""
        self.c.execute(query, (0, 0,))

    def insert_time(self, elapsed_time):
        self.c.execute("""UPDATE time_parameters set elapsed_time = {}""".format(elapsed_time))

    def insert_last_iteration(self, last_i):
        self.c.execute("""UPDATE time_parameters set last_i = {}""".format(last_i))

    def create_product_plane0_tracking_table(self):
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS product_plane0_tracking
               (iteration int,
                product text,
                plane int,
                jmatpro_conc float,
                existing_conc float,
                diff_conc float,
                cells_conc float)"""
        )
        self.c.execute("PRAGMA table_info(product_plane0_tracking)")
        existing_cols = {row[1] for row in self.c.fetchall()}
        if "cells_conc" not in existing_cols:
            self.c.execute(
                "ALTER TABLE product_plane0_tracking ADD COLUMN cells_conc float"
            )
        if "plane" not in existing_cols:
            self.c.execute(
                "ALTER TABLE product_plane0_tracking ADD COLUMN plane integer NOT NULL DEFAULT 0"
            )

    def insert_product_plane0_tracking(self, tracking_data):
        if not tracking_data:
            return
        rows = []
        for key, values in tracking_data.items():
            if len(key) == 2:
                iteration, product = key
                plane = 0
            else:
                iteration, product, plane = key[0], key[1], int(key[2])
            if len(values) >= 4:
                jmatpro_conc, existing_conc, diff_conc, cells_conc = values[:4]
            else:
                jmatpro_conc, existing_conc, diff_conc = values
                cells_conc = 0.0
            rows.append((
                int(iteration),
                str(product),
                int(plane),
                float(jmatpro_conc),
                float(existing_conc),
                float(diff_conc),
                float(cells_conc),
            ))
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
        self.c.executemany(
            """INSERT INTO product_plane0_tracking
               (iteration, product, plane, jmatpro_conc, existing_conc, diff_conc, cells_conc)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    @staticmethod
    def to_tuple(points):
        points = points.tolist()
        return [(point[0], point[1], point[2]) for point in points]

