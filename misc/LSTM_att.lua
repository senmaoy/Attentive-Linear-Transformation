require 'nn'
require 'nngraph'
local LSTM = {}
function LSTM.lstm(lstm_size,vocab_size)
    -- Model parameters
    local feat_size = 4096-- opt.feat_size_att

    local rnn_size = lstm_size
    local input_size = 512-- word embeding size
    local att_hid_size = 512--opt.att_hid_size
    local output_size = vocab_size
 
    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local img = nn.Identity()()   -- batch * att_size * feat_size -- the image patches
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------ Attention part --------------------

    local img_i = nn.Linear(feat_size, att_hid_size)(img)     
    local rnn_i = nn.Linear(rnn_size, att_hid_size)(prev_h) 
    local in_gate = nn.CAddTable(){img_i, rnn_i} 
    in_gate = nn.Tanh()(in_gate)
    in_gate = nn.Linear(att_hid_size,feat_size)(in_gate)
    in_gate = nn.Sigmoid()(in_gate)
    local img_o = nn.Linear(feat_size, att_hid_size)(img)     
    local rnn_o = nn.Linear(rnn_size, att_hid_size)(prev_h) 
    local out_gate = nn.CAddTable(){img_o, rnn_o}   
    out_gate = nn.Tanh()(out_gate)
    out_gate = nn.Linear(att_hid_size,rnn_size)(out_gate)
    out_gate = nn.Sigmoid()(out_gate)  
    
    local img_in = nn.CMulTable(){in_gate,img}
    img_in = nn.Linear(feat_size,rnn_size)(img_in)

    local img_out = nn.CMulTable(){out_gate,img_in}
    
    -------------- End of attention part -----------
    
    --- Input to LSTM
    local att_add = nn.Linear(rnn_size, 4 * rnn_size)(img_out)   -- batch * (4*rnn_size) <- batch * feat_size

    ------------- LSTM main part --------------------
    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    
    -- test
    -- local prev_all_input_sums = nn.CAddTable()({i2h, h2h})
    -- local all_input_sums = nn.CAddTable()({prev_all_input_sums, att_add})

    local all_input_sums = nn.CAddTable()({i2h, h2h, att_add})

    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)

    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    
    -- set up the decoder
    local top_h = next_h
    top_h = nn.Dropout(0.5)(top_h):annotate{name='drop_final'} 
    local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
    local logsoft = nn.LogSoftMax()(proj)
    return nn.gModule({x,img, prev_c, prev_h}, {next_c, next_h,logsoft})
end
return LSTM