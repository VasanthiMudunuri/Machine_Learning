import numpy as np
import random
import matplotlib.pyplot as mplot

class HMM:
    outcomes = np.array([])
    def __init__(self):
        self.probabilityFair={'1':0.16667,'2':0.16667,'3':0.16666,'4':0.16667,'5':0.16667,'6':0.16666}
        self.probabilityLoaded={'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.5}
        self.transitionProbabilityFair={'F': 0.95,'L':0.05}
        self.transitionProbabilityLoaded={'L':0.90,'F':0.10}
        self.state='F'
    def transition(self,Outcomes):
        NoofOutcomes=len(Outcomes)  
        probabilityofFairstate=0.5
        probabilityofLoadedstate=0.5
        for i in range(0,NoofOutcomes):
            probabilityofFairstate=probabilityofFairstate*(self.transitionProbabilityFair['F']+self.transitionProbabilityFair['L'])*self.probabilityFair[Outcomes[i]]
            probabilityofLoadedstate=probabilityofLoadedstate*(self.transitionProbabilityLoaded['F']+self.transitionProbabilityFair['L'])*self.probabilityLoaded[Outcomes[i]]
        if self.state=='F'and probabilityofLoadedstate>probabilityofFairstate:
            self.state='L'
        elif self.state=='L' and probabilityofFairstate>probabilityofLoadedstate:
            self.state='F'            
    def sequence(self,N=1000):
        outputs='123456'
        states = np.array([])
        for i in range(N):
            HMM.outcomes=np.append(HMM.outcomes,random.choice(outputs))
            states=np.append(states,self.state)
            self.transition(HMM.outcomes)
        return ''.join(HMM.outcomes),''.join(states)
    states = ('F', 'L')
    end_state = 'L'
    observations = ('1','2','3','4','5','4','1','6')
    start_Probability = {'F': 0.5, 'L': 0.5}
    transition_Probability = {
    'F' : {'F': 0.95,'L':0.05},
    'L' : {'L':0.90,'F':0.10}}
    emission_Probability = {
    'F' : {'1':0.16667,'2':0.16667,'3':0.16666,'4':0.16667,'5':0.16667,'6':0.16666},
    'L' : {'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.5}}
    def forward_backward(observations, states, start_probability, transition_probability, emmission_probability, end_state):
        forward = []
        forward_previous = {}
        for i, observation in enumerate(observations):
            forward_current = {}
            for s in states:
                if i == 0:
                    forward_previous_sum = start_probability[s]
                else:
                    forward_previous_sum = sum(forward_previous[k]*transition_probability[k][s] for k in states)
                forward_current[s] = emmission_probability[s][observation] * forward_previous_sum
            forward.append(forward_current)
            forward_previous = forward_current
        Forward = sum(forward_current[k] * transition_probability[k][end_state] for k in states)
        backward = []
        backward_previous = {}
        for i, observation_reverse in enumerate(reversed(observations[1:]+(None,))):
            backward_current = {}
            for s in states:
                if i == 0:
                    backward_current[s] = transition_probability[s][end_state]
                else:
                    backward_current[s] = sum(transition_probability[s][l] * emmission_probability[l][observation_reverse] * backward_previous[l] for l in states)
            backward.insert(0,backward_current)
            backward_previous = backward_current
        Backward = sum(start_probability[l] * emmission_probability[l][observations[0]] * backward_current[l] for l in states)
        posterior = []
        for i in range(len(observations)):
            posterior.append({s: forward[i][s] * backward[i][s] / Forward for s in states})
        return forward, backward, posterior
    F,B,P= forward_backward(observations,states,start_Probability,transition_Probability,emission_Probability,end_state)
    print F,'\n\n',B,'\n\n',P
    print '\n\n'
if __name__ == '__main__':
    m = HMM()
    L = [m.sequence() for i in range(1)]
    for outputs,states in L:
        print '\n'
        print states + '\n' + outputs + '\n'            
       
        
        
            
        


