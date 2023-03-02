from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .maddpg_vdn_style_learner import MADDPG_VDN_Learner
from .maddpg_qmix_style_learner import MADDPG_QMIX_Learner
from .maddpg_non_monotonic_style_learner import MADDPG_NonMon_Learner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["maddpg_vdn_style_learner"] = MADDPG_VDN_Learner
REGISTRY["maddpg_qmix_style_learner"] = MADDPG_QMIX_Learner
REGISTRY["maddpg_non_monotonic_style_learner"] = MADDPG_NonMon_Learner

