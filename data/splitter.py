import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


class Splitter:

    def __init__(self,metadata,cfg) -> None:
        self.metadata         = metadata
        self.cfg              = cfg
        if self.cfg.seed == 'random':
            self.random       = np.random.default_rng()
            self.seed         = None
        else:
            self.seed         = int(self.cfg.seed)
            self.random       = np.random.default_rng(self.seed) 
            
    def __ri(self,df):
        return df.reset_index(drop=True)

    def _split_lu(self,train):
        fraction      = self.cfg.labeled_fraction
        if isinstance(fraction,float):
            tl        = train.sample(frac=fraction,random_state=self.seed)
        elif isinstance(fraction,int):
            tl        = train.sample(n=fraction,random_state=self.seed)

        if fraction == 1.0 or fraction == len(tl):
            tu        = train.sample(n=1,random_state=self.seed)
        else:
            tu        = train.drop(tl.index)
        return self.__ri(tl),self.__ri(tu)

    def _generate_split(self,tvl_df):
        if self.cfg.fold_strategy_rule == 'default':
            val           = tvl_df.sample(frac=0.2,random_state=self.seed)
            train         = tvl_df.drop(val.index)
            return self.__ri(train),self.__ri(val)
        else:
            pattern1      = r"\s*fold_num=\{(\d+)\}\s*,\s*num_folds=\{(\d+)\}\s*,\s*strategy=\{([^,}]+)(?:,group=\{([^']+)\})?\}\s*"
            match1        = re.fullmatch(pattern1, self.cfg.fold_strategy_rule)
            if match1:
                fold_num  = int(match1.group(1))
                num_folds = int(match1.group(2))
                strategy  = match1.group(3)
                group_col = match1.group(4)
                if strategy == 'kfold':
                    if num_folds > len(tvl_df):
                        tvl_df = self.__ri(tvl_df.sample(n=num_folds,replace=True,random_state=self.seed))
                    kf = KFold(n_splits=num_folds,shuffle=True,random_state=self.seed)
                    splits = kf.split(tvl_df.index)
                elif strategy == 'gkfold':
                    if num_folds >  len(tvl_df[group_col].unique()):
                        if num_folds > len(tvl_df):
                            tvl_df = self.__ri(tvl_df.sample(n=num_folds,replace=True,random_state=self.seed))
                        else:
                            tvl_df[group_col] = np.random.choice(num_folds,size=len(tvl_df),random_state=self.random)
                    kf = GroupKFold(n_splits=num_folds)
                    splits = kf.split(tvl_df.index, groups=tvl_df[group_col])
                for i, (t_idx, v_idx) in enumerate(splits):
                    if i == fold_num:
                        train_l = self.__ri(tvl_df.iloc[t_idx])
                        val_l   = self.__ri(tvl_df.iloc[v_idx])
                        return train_l,val_l
                    
    
    def filterids(self,df,input_str):
        pattern1     = r"random=\{(\d+)\}"
        pattern2     = r"query=\{([^\"]+)\}"
        pattern3     = r"random=\{(\d+)\}\s*,\s*query=\{([^\"]+)\}"
        pattern4     = r"random=\{(\d+)\}\s*,\s*colname=\{([^\"]+)\}"

        match1       = re.fullmatch(pattern1, input_str)
        match2       = re.fullmatch(pattern2, input_str)
        match3       = re.fullmatch(pattern3, input_str)
        match4       = re.fullmatch(pattern4, input_str)

        if match3:
            n_items     = int(match3.group(1))
            query       = match3.group(2)
            df          = df.query(query)
            df          = df.sample(n=n_items,random_state=self.seed)
            return self.__ri(df)
        elif match4:
            n_items     = int(match4.group(1))
            colname     = match4.group(2)           
            uitems      = np.random.choice(
                                list(df[colname].unique()),
                                size=n_items,
                                replace=False
                            )
            return self.__ri(df.loc[df[colname].isin(uitems)])
        elif match1:
            n_items     = int(match1.group(1))
            return self.__ri(df.sample(n=n_items,random_state=self.seed))
        elif match2:
            query       = match2.group(1)
            return self.__ri(df.query(query))
        else:
            raise ValueError('invalid pattern for filtering tval and test')

    def train_test_split(self,metadata):
        pattern3     = r"size=\{((0?\.[0-9]+)|(all))\}"
        pattern4     = r"key=\{([^\",']+)\}\s*,values=\{([^\"]+)\}"
        if self.cfg.test_split_rule == 'default':
            self.cfg.test_split_rule='size={0.3}'    
        test_split_rule = self.cfg.test_split_rule
        match3       = re.fullmatch(pattern3, test_split_rule)
        match4       = re.fullmatch(pattern4, test_split_rule)
        if match3:
            test_size= match3.group(1)
            if test_size == 'all':
                test     = metadata
                tval     = metadata.sample(n=1,random_state=self.seed)
            else:
                test_size = float(test_size)
                test     = metadata.sample(frac=test_size,random_state=self.seed)
                tval     = metadata.drop(test.index)
        elif match4:
            key      = match4.group(1)
            values   = match4.group(2).replace("'", "").replace(" ", "").split(',')
            test     = metadata[metadata[key].isin(values)]
            tval     = metadata.drop(test.index)
        else:
            raise ValueError('Invalid or No test_split configuration provided for this dataset')
        test         = self.__ri(test)
        tval         = self.__ri(tval)
        return tval,test


    def get_split(self):

        if self.cfg.filter_rule == 'no_filter':
            metadata = self.metadata
        else:
            metadata = self.filterids(self.metadata,self.cfg.filter_rule)

        if self.cfg.split_name == 'all':
            return metadata

        elif self.cfg.split_name == 'test':
            tval,test       = self.train_test_split(metadata)
            return test
        elif self.cfg.split_name == 'val':
            tval,test       = self.train_test_split(metadata)
            train,val       = self._generate_split(tval)
            return val
        elif self.cfg.split_name == 'train':
            tval,test       = self.train_test_split(metadata)
            train,val       = self._generate_split(tval)
            return train
        elif self.cfg.split_name  == 'train_l':
            tval,test       = self.train_test_split(metadata)
            train,val       = self._generate_split(tval)
            train_l,train_u = self._split_lu(train)
            return train_l
        elif self.cfg.split_name  == 'train_u':
            tval,test       = self.train_test_split(metadata)
            train,val       = self._generate_split(tval)
            train_l,train_u = self._split_lu(train)
            return train_u