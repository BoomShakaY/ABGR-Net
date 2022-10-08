import torch
import torch.nn as nn


"""
PyTorch modules for dealing with graphs.

"""


def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)

class FusionLayer(nn.Module):
  def __init__(self, img_dim, word_dim, activation, method='concatenate'):
    super(FusionLayer, self).__init__()

    if method =='add':
      self.W_i = nn.Linear(img_dim, img_dim, bias=False)
      self.W_t = nn.Linear(word_dim, img_dim, bias=False)
    if method == 'concatenate':
      self.mlp = nn.Linear(img_dim+word_dim, img_dim) 
    self.method = method 


    if activation =='tanh':
      self.act =nn.Tanh()
    elif activation =='relu':
      self.act = nn.ReLU()
    else:
      self.act = None
    
    
  def forward(self, vecs, labels):
    if self.method =='add':
      output = self.W_i(vecs)+self.W_t(labels)
      if self.act is not None:
        output = self.act(output)
    if self.method == 'concatenate':
      output = self.mlp(torch.cat([vecs, labels], dim=1))
      if self.act is not None:
        output = self.act(output)
    return output


class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512, alpha=1.0,
               mlp_normalization='none', activation=None):
    super(GraphTripleConv, self).__init__()
    
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.alpha = alpha
    self.net_r = nn.Sequential(nn.Linear(3*input_dim, output_dim))
    self.net_o = nn.Sequential(nn.Linear(input_dim, output_dim))
                                
    if activation == 'tanh':
      self.net_r_act = nn.Tanh()
      self.net_o_act  = nn.Tanh()
                                
    elif activation == 'leakyrelu':
      self.net_r_act = nn.LeakyReLU()
      self.net_o_act  = nn.LeakyReLU()
    else:
      print("no such activate function")
      self.net_r_act = None
      self.net_o_act  = None
                               


  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype= obj_vecs.dtype
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, Dout = self.input_dim, self.output_dim
    
    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()
    
    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]
    
    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new pred vecs; shape is (T,  Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_p_vecs = self.net_r(cur_t_vecs)
    if self.net_r_act is not None:
      new_p_vecs = self.net_r_act(new_p_vecs)


    new_obj_vecs = self.net_o(obj_vecs)
    if self.net_o_act is not None:
      new_obj_vecs = self.net_o_act(new_obj_vecs)

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim, num_layers=5, hidden_dim=512, alpha=1.0,
               mlp_normalization='none', activation=None):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim': input_dim,
      'hidden_dim': hidden_dim,
      'alpha':alpha,
      'mlp_normalization': mlp_normalization,
      'activation':activation,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
    return obj_vecs, pred_vecs


