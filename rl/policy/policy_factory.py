from simulator.policy.policy_factory import policy_factory
from rl.policy.cadrl import CADRL
from rl.policy.lstm_rl import LstmRL
from rl.policy.sarl import SARL
from rl.policy.sail import SAIL

policy_factory["cadrl"] = CADRL
policy_factory["lstm_rl"] = LstmRL
policy_factory["sarl"] = SARL
policy_factory["sail"] = SAIL
