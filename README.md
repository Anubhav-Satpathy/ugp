This notebook guides you through different approaches for polyphonic audio melody extraction

First a pure signal proccesing approach is used (F0_tracking() of LibFMP library). The salience representation is used for getting most salient harmonic at each frame. This is followed by applying temporal continuity constraints such as transition matrix. The results of F0_tracking() on 16 audio files of MIR1K dataset can be found in the .txt file

Next the performance of different deep learning approaches for polyphonic audio melody extraction are compared. Precisely 3 types of architectures are used - LSTM, Bi-LSTM and Deep Bi-LSTM (2 layered). This is beyond the scope of MLSP (EE698V) as sequence-to-sequence models such as LSTM, Bi-LSTM and Deep Bi-LSTM are not covered in the course. The comparison of different models can be found in the notebook.

Finally the trained model is saved. Since Deep Bi-LSTM is the best performing, it is saved for future use. This model has also been loaded in the demo notebook. Run the full demo notebook in order to see the demo.
As the saved model was a large file (~ 484 MB) so it could not be uploaded. Please run the deep learning notebook. It will automatically save the deep bilstm model in the current folder. After that it can be loaded in the demo notebook
