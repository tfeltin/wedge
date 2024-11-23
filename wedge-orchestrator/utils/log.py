from time import sleep
from threading import Thread

from utils.utils import print_real_times, predicted_times

class Logger:
    def __init__(self, placement, dnnconf, netconf, sync_agent, config):
        self.config = config
        self.sync_agent = sync_agent
        self.dnnconf = dnnconf
        self.netconf = netconf
        self.placement = placement
        self.run_flag = False
        self.logging_thread = Thread(target=self.run, daemon=True)
    
    def run(self):
        pred_times = predicted_times(self.placement, self.dnnconf, self.netconf)
        # KEEP TRACK OF WORKER PERFORMANCES
        while self.run_flag:
            self.sync_agent.pull_all_times()
            try:
                print_real_times(self.sync_agent.state, self.placement, pred_times)
                sleep(1)
            except:
                sleep(1)
         
    def start(self):
        self.run_flag = True
        self.logging_thread.start()
    
    def stop(self):
        self.run_flag = False