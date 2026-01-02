
Write a chess decoder transformer model.

It will take tokens for input and have multiple heads.

Each chess positions will be encoded as multiple tokens, showing the pieces on the board.

For example, a basic chess position could be encoded as: 


<start_pos>, <white_king_g1>, <white_pawn_g2>, <black_king_g8>,<end_pos>,<castling_rights_token>,<side_to_move_token> for a basic chess position.

And for example in this position, if white plays the move g2g4, we would append the token <g2g4> to the sequence, and then output the new position 

leading to : 

<start_pos>, <white_king_g1>, <white_pawn_g2>, <black_king_g8>,<end_pos>,<castling_rights_token>,<side_to_move_token>,<g2g4>,<start_pos>,<white_king_g1>, <white_pawn_g4>, <black_king_g8>,<end_pos>,<castling_rights_token>,<side_to_move_token>


Use the classic multihead self attention from torch.

The model will have multiple output heads, one to predict the next token, which will be the policy head, but also another one that will predict the value of the position, from the perspective of the side to move, in the form of a (win,draw,lose) vector (Which we will probe at the right token, since it doesn't make sense to probe it mid-position)


We will use a classic RoPe encoding.

- Size the transformer properly. I aim to have a context window of 2048 tokens.

- Complete the vocabulary with the necessary tokens.

- Implement the transformer decoder model, following all classic transformer architecture. Use torch and torchtune. Implement inside src/

- Import the model in a separate script, and show its size breakdown, from head sizes, to embedding sizes, and the transformer layer sizes.


- Try to call the model with a batch size of 16 and a sequence length of 2048 tokens, and show the memory usage.

The code should be of high quality, with as little verbosity as necessary. No try except. No comments. No print statements. No debug code. 

- Use uv to run python. Install with "uv add ..." the necessary missing dependencies if needed.
