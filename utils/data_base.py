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
        self.save_pickled_config_to_db()

    def save_pickled_config_to_db(self):
        dict_config_to_pickle = get_static_vars_dict(Config)
        pickled_instance = pickle.dumps(dict_config_to_pickle)
        self.c.execute('''CREATE TABLE IF NOT EXISTS PickledConfig (pickled_data BLOB)''')
        self.c.execute("INSERT INTO PickledConfig (pickled_data) VALUES (?)", (pickled_instance,))
        self.conn.commit()

    def insert_particle_data(self, particle_type, iteration, data):
        """Particle types allowed: primary_oxidant, secondary_oxidant, primary_active, secondary_active,
        primary_product, secondary_product, ternary_product, quaternary_product"""
        query = """CREATE TABLE {}_iter_{} (z int, y int, x int)""".format(particle_type, str(iteration))
        self.c.execute(query)
        query = "INSERT INTO {}_iter_{} VALUES(?, ?, ?);".format(particle_type, str(iteration))
        data = data.transpose()
        data = self.to_tuple(data)
        self.c.executemany(query, data)

    def create_precipitation_front_table(self):
        self.c.execute(f"""CREATE TABLE precip_front_p (sqrt_time int, position int)""")
        if Config.ACTIVES.SECONDARY_EXISTENCE:
            self.c.execute("""CREATE TABLE precip_front_s (sqrt_time int, position int)""")

    def insert_precipitation_front(self, sqrt_time, position, sign):
        """sign p for primary product
            sign s for secondary product"""
        self.c.execute("INSERT INTO precip_front_{} VALUES ({}, {})".format(sign, sqrt_time, position))

    def create_time_parameters_table(self):
        self.c.execute("""CREATE TABLE time_parameters (last_i int, elapsed_time float)""")

        query = """INSERT INTO time_parameters VALUES(?, ?);"""
        self.c.execute(query, (0, 0,))

    def insert_time(self, elapsed_time):
        self.c.execute("""UPDATE time_parameters set elapsed_time = {}""".format(elapsed_time))

    def insert_last_iteration(self, last_i):
        self.c.execute("""UPDATE time_parameters set last_i = {}""".format(last_i))

    @staticmethod
    def to_tuple(points):
        points = points.tolist()
        return [(point[0], point[1], point[2]) for point in points]

