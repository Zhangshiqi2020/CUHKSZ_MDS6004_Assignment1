'''
Code referenced from:
https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
'''

import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO # Python 3.x

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)

        # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        # with self.writer.as_default():
        #         tf.summary.scalar(tag， summary, step=step)
        #         self.writer.flush()

        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

        
'''
Code referenced from: 
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
            input = input.transpose(1,2) # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()


import torch.nn as nn
import torch.nn.functional as F

class KDLoss:
    """
    A class to compute the knowledge distillation (KD) loss between student and teacher networks.
    This class allows the use of an instance like a function.
    """
    
    def __init__(self):
        """
        Initialize the KDLoss class. No parameters are required for initialization.
        """
        pass
    
    def __call__(self, student_outputs, teacher_outputs, T):
        """
        Compute the KL divergence loss between the softened outputs of the student and teacher networks.
        
        Args:
            student_outputs (Tensor): The logits from the student network.
            teacher_outputs (Tensor): The logits from the teacher network.
            T (float): Temperature to soften the probabilities.
        
        Returns:
            KD_loss (Tensor): The computed KL divergence loss.
        """
        # Apply temperature scaling and compute log probabilities for the student outputs
        student_log_probs = F.log_softmax(student_outputs / T, dim=1)
        
        # Apply temperature scaling and compute soft probabilities for the teacher outputs
        teacher_probs = F.softmax(teacher_outputs / T, dim=1)
        
        # Compute the KL Divergence loss between the student and teacher probabilities
        KD_loss = nn.KLDivLoss()(student_log_probs, teacher_probs) * (T ** 2)
        
        return KD_loss
