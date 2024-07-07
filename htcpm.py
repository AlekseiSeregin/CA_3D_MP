from engine import *
import traceback
from configuration import Config

if __name__ == '__main__':

    Config.COMMENT = """ 
    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_gb
    seeds = G_100

    Script name: htcpm.py
    For the GB-video.
"""

    eng = CellularAutomata()

    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_gb

    try:
        eng.simulation()
    finally:
        try:
            if not Config.SAVE_WHOLE:
                eng.save_results()

        except (Exception,):
            print("Not SAVED!")
        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()
