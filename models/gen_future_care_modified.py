# package imports modified 2/6
import numpy as np
import pandas as pd
import boto3
from io import BytesIO
client = boto3.client('s3')
import joblib
from itertools import combinations
import requests
#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
#import warnings
#warnings.filterwarnings("ignore")
#import cPickle
#from dateutil.relativedelta import relativedelta
#pd.set_option('display.max_colwidth', -1)


from sqlalchemy import create_engine
from sqlalchemy import types
import mysql.connector
# connects to dev; need information for production. I assume tables/column names the same
engine = create_engine(
    'mysql+mysqlconnector://root:Chetu#123@analyticpowered.cqwjpe7miwnj.us-west-2.rds.amazonaws.com/carebridge_dev')

class GenerateProFutureCare:


    def __init__(self, ref_id):#, claim_type

        self.ref_id = str(ref_id)
        self.drugs = self.get_medication_inputs('claim_claimdrug')
        self.reserve = True

    def get_medication_inputs(self, table):
        """
        Loads medication information input by users or else returns an empty dataframe. Needed data stored in
        claim_claimdrug and claim_futurecaredrug
        """
        try:
            #query = ('select drug, strength, count, route, is_off_label, ref_id from ' + table + ' where ref_id ='
            #                                                                                     ' ' + self.ref_id)
            #from_claimdrug = pd.DataFrame(list(ClaimDrug.objects.values('drug', 'strength', 'count', 'route',
            #                                                            'is_off_label', 'ref_id')
            #                                   .filter(ref_id=self.ref_id)))

            query = ('select drug, strength, count, route, is_off_label, ref_id from '+table+' where ref_id ='
                    ' '+self.ref_id)
            from_claimdrug = pd.read_sql(query,con=engine)

            # if blank, fill with appropriate value and change dtypes
            from_claimdrug = self.fill_blanks_change_dtype(from_claimdrug, col='count', fill_value=1)
            # from_claimdrug = self.fill_blanks_change_dtype(from_claimdrug, col='route',fill_value=np.nan,
            #                                               numeric_dtype=False)
            from_claimdrug['refill'] = 12
            from_claimdrug['every_x_year'] = 1
            # concat vertically
            drugs = from_claimdrug

            # if claimdrug has more values than futurecaredrug, remove excess
            drugs = drugs[drugs.refill.notnull()]

            return drugs
        except:
            return pd.DataFrame()



    def get_claim_data_class(self):
        """
                Given a ref_id, extracts all relevant medication inputs (compensable diagnoses, claim type, drug, strength,
                    route, is_off_label, Every X Years, and refills) and transforms them into priced, generic alternatives
                """
        self.drugs = self.get_medication_inputs('claim_claimdrug')
        # if ref_id has medications attached to claim
        if len(self.drugs) > 0:
            self.medications = self.get_medications()
        else:
            # return an empty dataframe
            self.medications = pd.DataFrame()
        return self

    def load_from_s3(self, key=None, dtypes={}, bucket='dimitricbi'):
        """
        Streamlines loading dataframes from csv files in s3
        :param key:
        :param dtypes:
        :param bucket:
        :return:
        """
       # if self.env in ['development', 'production']:
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(BytesIO(obj['Body'].read()), dtype=dtypes)
        #else:
        #    file_name = key.split('/')[-1]
        #    df = pd.read_csv(settings.BASE_DIR + '/claim/files/' + file_name, dtype=dtypes)
        return df

    def fill_blanks_change_dtype(self, df, col=None, fill_value=None, numeric_dtype=True):
        """
        Checks for and fills empty strings, then converts column dtype if desired
        """
        # if empty string, fill with prescribed value
        mask = df[col] == ''
        df.loc[df[mask].index, col] = fill_value
        # if column is numeric, chage dtype to float
        if numeric_dtype == True:
            df[col] = df[col].astype(float)
        return df



    def format_icds(self, db):
        """
        Obtains compensable diagnoses from claim_claimdiag. Cleans but does not crosswalk icds. Only called for MSAs.
        """
        self.cleaned_codes = []
        # added 7/8
        query = 'select icd from ' + db + ' where ref_id = ' + self.ref_id
        df = pd.DataFrame(list(ClaimDiag.objects.values('icd').filter(ref_id=self.ref_id)))
        for code in list(df.icd.unique()):

            # standardizing format of dx codes
            code = str(code).replace('.', '')
            code = code.upper()
            # crop code first to get rid of sequela
            if len(code) == 7:
                code = code[:6]
            self.cleaned_codes.append(code)
        return

    def is_reserve(self):
        """
        Determines whether claim is an MSA or Reserve by querying claim_claimantinfo
        """
        #
        query = 'select claim_type from claim_claimantinfo where ref_id =' + self.ref_id
        df = pd.DataFrame(list(ClaimantInfo.objects.values('claim_type').filter(pk=self.ref_id)))
        # uppercase
        df.claim_type = df.claim_type.str.upper()
        if len(df[df.claim_type.str.contains('MSA')]) > 0:
            reserve = False
        else:
            reserve = True
        return reserve

    # get generic equivalent for drug provided
    def get_ndc(self, drug):
        """
        Obtains corresponding ndc where drug name and other information is present
        """
        ndc_val = pd.DataFrame()

        # demo_df now limited to get_ndc
        demo_df = self.drugs[self.drugs.drug == drug].copy()

        # change blank values to nulls and uppercase remaining
        for col in ['drug', 'strength', 'route']:
            # without mask, null values will capitalized and become strings
            mask = demo_df[col].notnull()
            demo_df.loc[demo_df[mask].index, col] = demo_df.loc[demo_df[mask].index, col].map(
                lambda x: np.nan if x == '' else str(x).upper())

        # identify the notnull demo_df columns
        drug_combo = demo_df[['drug', 'strength', 'route']].copy().dropna(
            axis=1)

        # change the column names to mirror ndc_df 
        drug_combo.rename(columns={'drug': 'Claim_Drug', 'strength': 'Claim_Strength',
                                   'route': 'Claim_Route'}, inplace=True)

        # create list of columns to join on
        join_cols = drug_combo.columns

        # need to at least have drug name to obtain ndc
        if 'Claim_Drug' in join_cols:

            # if match identified, take cheapeast alternative
            ndc_val = self.ndc_df.merge(drug_combo, on=list(join_cols)).sort_values('Price Per Pill').head(1)

            # testing shows different strength values can mess up join, so if no result on first try, just use drug
            if len(ndc_val) == 0:
                ndc_val = self.ndc_df.merge(drug_combo[['Claim_Drug']], on='Claim_Drug').sort_values(
                    'Price Per Pill').head(1)

        # if ndc_val now a df and not an empty string
        if len(ndc_val) > 0:
            ndc_val['Quantity'] = demo_df['count'].values[0]
            ndc_val['PartD'] = demo_df['is_off_label'].values[0]
            ndc_val['Refills'] = demo_df['refill'].values[0]
            ndc_val['annual'] = demo_df['every_x_year'].values[0]
            return ndc_val
        # else return null
        else:
            return pd.DataFrame()

    def is_off_label(self, gfc):
        """
        takes a generic formulary code and determines if drug is off label
        """
        off_label = 0
        # if gfc in list of off_label gfcs
        if gfc in self.off_label_df.gfc.unique():
            # grab generic cross reference
            gcr = self.off_label_df[self.off_label_df.gfc == gfc].gcr.unique()[0]
            # assume drug is off_label
            off_label = 1
            # but loop through icds
            for icd in self.cleaned_codes:
                # and if dx_code is an acceptable dx
                if icd in self.gcr_icd[self.gcr_icd.gcr == gcr].icd.unique():
                    # set off_label to 0
                    off_label = 0
        return off_label

    def refills_off_label_msa(self, medications):
        """
        Evaluated whether drug in MSA is off label and applies refills and future years values
        """
        # create list to flag if rx off label and iterate through gfcs
        if self.reserve == False:

            # only need to format_icds to check if drug off label
            self.format_icds('claim_claimdiag')

            off_label_list = []

            for gfc in medications.gfc.values:
                # added do is_off_label can be called on a one off basis
                off_label_list.append(self.is_off_label(gfc))
            medications['off_label'] = off_label_list
            # if nurse indicated drug is off label, change value to off label, changed 4/17
            mask = (medications['PartD'] == -1) & (medications['off_label'] == 0)
            medications.loc[medications[mask].index, 'off_label'] = 1

        return medications

    def format_medications(self, medications):
        '''
        Formats medications to match desired display in report
        '''

        # use unrounded unit price value to find Annual price

        medications['Total Annual'] = (medications['Quantity'] * medications[
            'Price Per Pill'] * medications['Refills']) / medications['annual']

        # round values
        medications['Price Per Pill'] = medications['Price Per Pill'].round(4)
        medications['Total Annual'] = medications['Total Annual'].round(2)

        # monthly cost is over 12 months, but prescription might not be filled monthly
        medications['Monthly Cost'] = medications['Total Annual'] / 12
        medications['Monthly Cost'] = medications['Monthly Cost'].round(2)

        # these are the columns to display in order; header names to change
        cols = ['most_generic', 'NDC', 'Claim_Strength', 'Price Per Pill', 'Quantity', 'Refills', 'annual',
                'Monthly Cost', 'Total Annual']

        if self.reserve == False:
            cols.append('off_label')

        # changes to headers requested by staff
        medications = medications[cols].rename(
            columns={'most_generic': 'Medication', 'NDC': 'NDC Code', 'Claim_Strength':
                'Strength', 'Price Per Pill': 'Unit Price (per pill/ml/gm)',
                     'annual': 'Every X Yrs.'})

        # For AP MSA, add a Total column, where the total is Total Annual * Life Expectancy
        # If off_label == 1 AND the claim type is an MSA, the cost for the off label drug is represented in the
        # Off Label Drugs line; all other medications are allocated to Part D.
        # remove off_label flag from display in report
        # add sum of Total Lifetime as Subtotal to new row in last two columns of report
        return medications

    def get_medications(self):
        """
        Runs all methods required to build medications table
        """
        medications = pd.DataFrame()

        # load and filter ndc_df
        self.ndc_df = self.load_from_s3(key='Bin_Benchmark/product_to_generic_msa_reserve_0327.csv',
                                        dtypes={'NDC': 'str', 'tcc': 'str', 'gfc': 'str'})
        if self.reserve == True:
            self.ndc_df = self.ndc_df[self.ndc_df['reserve'] == 1]
        else:
            self.ndc_df = self.ndc_df[self.ndc_df['reserve'] == 0]

            # off_label_df and grc_icd now attributes to work with is_off_label,
            # loaded if refills_off_label_msa not called
            self.off_label_df = self.load_from_s3(key='Bin_Benchmark/off_label_drugs.csv',
                                                  dtypes={'gcr': 'str', 'gfc': 'str', 'ndc': 'str'})

            self.gcr_icd = self.load_from_s3(key='Bin_Benchmark/off_label_gcr_to_icd.csv',
                                             dtypes={'gcr': 'str', 'icd': 'str'})

        # obtain generic equivalent for drug name/strength/route provided
        for drug in self.drugs.drug.unique():
            medications = pd.concat([medications, self.get_ndc(drug)], ignore_index=True).drop_duplicates()
        if len(medications) > 0:
            medications = self.refills_off_label_msa(medications)
            medications = self.format_medications(medications)
            # for MSA, default of annual recurrence
            medications['Every X Yrs.'] = 1
            medications['Total Annual'] = (medications['Quantity'] * medications[
                'Unit Price (per pill/ml/gm)'] * medications['Refills']) / medications['Every X Yrs.']
            medications['Total Annual'] = medications['Total Annual'].round(2)
            # monthly cost is over 12 months, but prescription might not be filled monthly
            medications['Monthly Cost'] = medications['Total Annual'] / 12
            medications['Monthly Cost'] = medications['Monthly Cost'].round(2)
            # to handle model output handing in UI
            medications['Total'] = 0

        return medications

    def get_single_ndc(self, medication='', strength=''):
        """
        take a medication and strength (if available) as input, output NDC and off label determination
        """
        # load and filter ndc_df
        self.ndc_df = self.load_from_s3(key='Bin_Benchmark/product_to_generic_msa_reserve_0327.csv',
                                        dtypes={'NDC': 'str', 'tcc': 'str', 'gfc': 'str'})
        if self.reserve == True:
            self.ndc_df = self.ndc_df[self.ndc_df['reserve'] == 1]
        else:
            self.ndc_df = self.ndc_df[self.ndc_df['reserve'] == 0]

            # off_label_df and grc_icd now attributes to work with is_off_label,
            # loaded if refills_off_label_msa not called
            self.off_label_df = self.load_from_s3(key='Bin_Benchmark/off_label_drugs.csv',
                                                  dtypes={'gcr': 'str', 'gfc': 'str', 'ndc': 'str'})

            self.gcr_icd = self.load_from_s3(key='Bin_Benchmark/off_label_gcr_to_icd.csv',
                                             dtypes={'gcr': 'str', 'icd': 'str'})
        if len(strength) > 0:
            mask = (self.ndc_df.Claim_Drug == medication) & (self.ndc_df.Claim_Strength == strength)
        else:
            mask = (self.ndc_df.Claim_Drug == medication)

        # get appropriate mapping for MSA or Reserve, then take the cheapest
        # potential here for conflict where cheapest applies to different route than desired
        # nurse also unable to manually mark if drug off label here

        try:
            generic_ndc = self.ndc_df[mask].sort_values('Price Per Pill')['NDC'].values[0]
        except:
            return 'No Match Found'

        # if MSA, determine if drug is off label
        if self.reserve == False:
            # get gfc to run is_off_label
            generic_ndc_gfc = self.ndc_df[mask].sort_values('Price Per Pill')['gfc'].values[0]
            # value off 1 represents off_label
            self.single_off_label = self.is_off_label(generic_ndc_gfc)

        return generic_ndc
