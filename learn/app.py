from madrona_bots import ScriptBotsViewer

def void_function():
    pass

def train_step(sim_mgr):
    sim_mgr.step()
    print("Just stepped!")

def main():
    # First 4 parameters are the same as for SimManager.
    # The second 2 are the window width and height.
    viewer_app = ScriptBotsViewer(0, 4, 0, 16, 1375, 768)

    sim_mgr = viewer_app.get_sim_mgr()

    viewer_app.loop(lambda: train_step(sim_mgr))

if __name__ == "__main__":
    main()
