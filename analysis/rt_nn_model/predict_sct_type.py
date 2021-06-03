import pickle
import datetime

import keras
import numpy
import pandas
from sklearn import preprocessing

import pydarn


class PredctSctr(object):

    def __init__(self, model_name,\
             scaler_name, inp_df, rad="bks"):
        
        # load the model and the scaler
        self.model = keras.models.load_model(model_name, compile=False)
        with open(scaler_name, "rb") as f:
            self.scaler = pickle.load(f)

        # pred attributes
        self.inp_df = inp_df
        
        # get radar params
        self.hdw = pydarn.read_hdw_file(rad)
        self.boresite = self.hdw.boresight
        self.nbeams = self.hdw.beams
        self.ngates = self.hdw.gates
        self.txlat = self.hdw.geographic.lat
        self.txlon = self.hdw.geographic.lon
#         self.beam_sep = self.hdw.beam_seperation
        self.offset = self.nbeams/2. - 0.5
            
    def predict(self, batch_size=32):
        # calculate other features!
        
        inp_trn_params = [ "gate", "freq", "azim", "month_sine",\
                    "month_cosine", "time_sine", "time_cosine",\
                    "ap", "f107" ]
        
        if "azim" not in self.inp_df.columns:
            self.inp_df["azim"] = self.inp_df["beam"]
        
        self.inp_df["month"] = [x.month for x in self.inp_df["date"]]
        self.inp_df["hour"] = [x.hour for x in self.inp_df["date"]]
        self.inp_df["minute"] = [x.minute for x in self.inp_df["date"]]
        self.inp_df["minutes_in_day"] = self.inp_df["hour"]*60 + self.inp_df["minute"]
        
        # convert month to sine/cosine cyclical feature!
        self.inp_df["month_sine"] = numpy.sin(2*numpy.pi/12 * self.inp_df["month"])
        self.inp_df["month_cosine"] = numpy.cos(2*numpy.pi/12 * self.inp_df["month"])
        # convert time (hours and minutes) to sine/cosine cyclical feature!
        self.inp_df["time_sine"] = numpy.sin(2*numpy.pi/1440 * self.inp_df["minutes_in_day"])
        self.inp_df["time_cosine"] = numpy.cos(2*numpy.pi/1440 * self.inp_df["minutes_in_day"])
        
        print("scaled inputs!")
        input_data = self.scaler.transform(self.inp_df[inp_trn_params].values)
        print("predicting.....")
        y_pred = self.model.predict(input_data, batch_size=batch_size)
        #append the predictions to the df and return
        self.inp_df["iono_hf_sct_label_pred"] = numpy.round(y_pred[:,0],2)
        self.inp_df["iono_one_hf_sct_label_pred"] = numpy.round(y_pred[:,1],2)
        self.inp_df["gnd_one_sct_label_pred"] = numpy.round(y_pred[:,2],2)
        self.inp_df["gnd_two_sct_label_pred"] = numpy.round(y_pred[:,3],2)
        self.inp_df["no_sct_label_pred"] = numpy.round(y_pred[:,4],2)
        
        return self.inp_df

if __name__ == "__main__":
    
#     model_name = glob.glob(os.path.join("/home/data/rt", "weights.epoch_06"+"*hdf5"))[0]
    model_name = 'rt_nn_model/weights.epoch_06.val_loss_0.26.hdf5'
    scaler_name = "rt_nn_model/scaler.pck"
    inds_data = "rt_nn_model/inds_data.pkl"
    # setup an input df
    inp_df = pandas.DataFrame(
            {'gate': [x for x in range(1,75)],
             'date': [datetime.datetime(2017,5,1,1) for x in range(1,75)],
             'freq': [16 for x in range(1,75)],
             'azim': [7 for x in range(1,75)],
            })
    
    # read ap and f107
    inds_data = pandas.read_pickle(inds_data)
    inp_df = pandas.merge(inp_df, inds_data, on=["date"])
    
    print("###### input data ######")
    print(inp_df)
    print("###### input data ######")
    
    pred_obj = PredctSctr( model_name,\
                     scaler_name, inp_df)
    
    data_df = pred_obj.predict()
    print("###### output data ######")
    print(data_df[data_df["iono_sct_prb"] > 0.3])
    print("###### output data ######")
    
    
    
