CDF  �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-02-20T02:45:03Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.historical.none.r1i1p1f1   grid      7native atmosphere TL159 gaussian grid (160x320 latxlon)    
grid_label        gn     history      �Tue May 30 16:58:43 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/MRI-ESM2-0_r1i1p1f1/rsut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp126_v20190603.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_210101-230012.yearmean.mul.areacella_ssp126_v20190603.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/MRI-ESM2-0_r1i1p1f1/rsut_MRI-ESM2-0_r1i1p1f1_ssp126.mergetime.nc
Tue May 30 16:58:43 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190603.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/MRI-ESM2-0_r1i1p1f1/rsut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 07:27:54 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.fx.areacella.gn.v20190603/areacella_fx_MRI-ESM2-0_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190603.fldsum.nc
2019-02-20T02:45:03Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
Output from run-Dr060_historical_101 (sfc_avr_mon.ctl)    initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(14 December 2018) MD5:b2d32d1a0d9b196411429c8895329d8f      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/405a2b90-dfa1-4482-97dc-1f18ac879500      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   
   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   original_name         USWT   cell_measures         area: areacella    history       r2019-02-20T02:45:03Z altered by CMOR: replaced missing value flag (-9.99e+33) with standard missing value (1e+20).              �                Aq���   Aq��P   Aq�P   [0�hAq�6�   Aq�P   Aq��P   [0C�Aq���   Aq��P   Aq��P   [0�Aq��   Aq��P   Aq�dP   [1�VAq���   Aq�dP   Aq��P   [0�Aq���   Aq��P   Aq�FP   [0�6Aq�k�   Aq�FP   Aq��P   [0��Aq���   Aq��P   Aq�(P   [0��Aq�M�   Aq�(P   Aq��P   [0��Aq���   Aq��P   Aq�
P   [/�Aq�/�   Aq�
P   Aq�{P   [1�Aq���   Aq�{P   Aq��P   [1��Aq��   Aq��P   Aq�]P   [1��AqĂ�   Aq�]P   Aq��P   [0��Aq���   Aq��P   Aq�?P   [1�7Aq�d�   Aq�?P   Aq˰P   [1gAq���   Aq˰P   Aq�!P   [1<Aq�F�   Aq�!P   AqВP   [0��Aqз�   AqВP   Aq�P   [1(4Aq�(�   Aq�P   Aq�tP   [1~qAqՙ�   Aq�tP   Aq��P   [15HAq�
�   Aq��P   Aq�VP   [1t
Aq�{�   Aq�VP   Aq��P   [2 
Aq���   Aq��P   Aq�8P   [2S�Aq�]�   Aq�8P   Aq�P   [0�Aq���   Aq�P   Aq�P   [0��Aq�?�   Aq�P   Aq�P   [1FIAq��   Aq�P   Aq��P   [0��Aq�!�   Aq��P   Aq�mP   [1��Aq��   Aq�mP   Aq��P   [1RAq��   Aq��P   Aq�OP   [1hAq�t�   Aq�OP   Aq��P   [1xAq���   Aq��P   Aq�1P   [0��Aq�V�   Aq�1P   Aq��P   [3��Aq���   Aq��P   Aq�P   [5��Aq�8�   Aq�P   Aq��P   [2�SAq���   Aq��P   Aq��P   [2�Aq��   Aq��P   ArfP   [1�;Ar��   ArfP   Ar�P   [1�Ar��   Ar�P   ArHP   [0��Arm�   ArHP   Ar�P   [1lAr��   Ar�P   Ar*P   [1��ArO�   Ar*P   Ar�P   [1"|Ar��   Ar�P   ArP   [1H�Ar1�   ArP   Ar}P   [0��Ar��   Ar}P   Ar�P   [0�Ar�   Ar�P   Ar_P   [0��Ar��   Ar_P   Ar�P   [14�Ar��   Ar�P   ArAP   [0ۖArf�   ArAP   Ar�P   [24�Ar��   Ar�P   Ar!#P   [1)/Ar!H�   Ar!#P   Ar#�P   [0�	Ar#��   Ar#�P   Ar&P   [2�Ar&*�   Ar&P   Ar(vP   [3�2Ar(��   Ar(vP   Ar*�P   [1�LAr+�   Ar*�P   Ar-XP   [1��Ar-}�   Ar-XP   Ar/�P   [1�Ar/��   Ar/�P   Ar2:P   [2��Ar2_�   Ar2:P   Ar4�P   [1MXAr4��   Ar4�P   Ar7P   [1m�Ar7A�   Ar7P   Ar9�P   [1��Ar9��   Ar9�P   Ar;�P   [2UAr<#�   Ar;�P   Ar>oP   [3xdAr>��   Ar>oP   Ar@�P   [2��ArA�   Ar@�P   ArCQP   [2H�ArCv�   ArCQP   ArE�P   [1�ArE��   ArE�P   ArH3P   [1��ArHX�   ArH3P   ArJ�P   [2B�ArJ��   ArJ�P   ArMP   [2�ArM:�   ArMP   ArO�P   [2R{ArO��   ArO�P   ArQ�P   [2�gArR�   ArQ�P   ArThP   [2?ArT��   ArThP   ArV�P   [1�oArV��   ArV�P   ArYJP   [2�:ArYo�   ArYJP   Ar[�P   [2LdAr[��   Ar[�P   Ar^,P   [2˝Ar^Q�   Ar^,P   Ar`�P   [1�Ar`��   Ar`�P   ArcP   [2�Arc3�   ArcP   AreP   [2Are��   AreP   Arg�P   [2#%Arh�   Arg�P   ArjaP   [2R�Arj��   ArjaP   Arl�P   [2ZArl��   Arl�P   AroCP   [1�xAroh�   AroCP   Arq�P   [2/Arq��   Arq�P   Art%P   [1w�ArtJ�   Art%P   Arv�P   [1��Arv��   Arv�P   AryP   [2��Ary,�   AryP   Ar{xP   [1�Ar{��   Ar{xP   Ar}�P   [1��Ar~�   Ar}�P   Ar�ZP   [2V�Ar��   Ar�ZP   Ar��P   [1�Ar���   Ar��P   Ar�<P   [2k�Ar�a�   Ar�<P   Ar��P   [2�cAr���   Ar��P   Ar�P   [2��Ar�C�   Ar�P   Ar��P   [26�Ar���   Ar��P   Ar� P   [2��Ar�%�   Ar� P   Ar�qP   [1S�Ar���   Ar�qP   Ar��P   [1��Ar��   Ar��P   Ar�SP   [2c�Ar�x�   Ar�SP   Ar��P   [1��Ar���   Ar��P   Ar�5P   [2\Ar�Z�   Ar�5P   Ar��P   [2e�Ar���   Ar��P   Ar�P   [3�PAr�<�   Ar�P   Ar��P   [2��Ar���   Ar��P   Ar��P   [2��Ar��   Ar��P   Ar�jP   [3+?Ar���   Ar�jP   Ar��P   [3��Ar� �   Ar��P   Ar�LP   [2��Ar�q�   Ar�LP   Ar��P   [3-;Ar���   Ar��P   Ar�.P   [3�8Ar�S�   Ar�.P   Ar��P   [3�Ar���   Ar��P   Ar�P   [3�TAr�5�   Ar�P   Ar��P   [3�oAr���   Ar��P   Ar��P   [5ځAr��   Ar��P   Ar�cP   [6�Ar���   Ar�cP   Ar��P   [4�PAr���   Ar��P   Ar�EP   [4B]Ar�j�   Ar�EP   ArĶP   [4�Ar���   ArĶP   Ar�'P   [4aLAr�L�   Ar�'P   ArɘP   [4��Arɽ�   ArɘP   Ar�	P   [4oAr�.�   Ar�	P   Ar�zP   [4՚ArΟ�   Ar�zP   Ar��P   [5*�Ar��   Ar��P   Ar�\P   [4��ArӁ�   Ar�\P   Ar��P   [3��Ar���   Ar��P   Ar�>P   [6@�Ar�c�   Ar�>P   ArگP   [5c�Ar���   ArگP   Ar� P   [4��Ar�E�   Ar� P   ArߑP   [3��Ar߶�   ArߑP   Ar�P   [5�RAr�'�   Ar�P   Ar�sP   [4��Ar��   Ar�sP   Ar��P   [5��Ar�	�   Ar��P   Ar�UP   [5LAr�z�   Ar�UP   Ar��P   [7��Ar���   Ar��P   Ar�7P   [5��Ar�\�   Ar�7P   Ar�P   [4f�Ar���   Ar�P   Ar�P   [6G9Ar�>�   Ar�P   Ar��P   [4��Ar���   Ar��P   Ar��P   [5�gAr� �   Ar��P   Ar�lP   [5�Ar���   Ar�lP   Ar��P   [4�~Ar��   Ar��P   Ar�NP   [7��Ar�s�   Ar�NP   As�P   [9�ZAs��   As�P   As0P   [7#AsU�   As0P   As�P   [4�LAs��   As�P   As	P   [6#As	7�   As	P   As�P   [51qAs��   As�P   As�P   [4b�As�   As�P   AseP   [4��As��   AseP   As�P   [4!�As��   As�P   AsGP   [3�DAsl�   AsGP   As�P   [5As��   As�P   As)P   [4�.AsN�   As)P   As�P   [4b)As��   As�P   AsP   [3��As0�   AsP   As!|P   [4��As!��   As!|P   As#�P   [4��As$�   As#�P   As&^P   [4tAs&��   As&^P   As(�P   [4��As(��   As(�P   As+@P   [4�As+e�   As+@P   As-�P   [4R�As-��   As-�P   As0"P   [3�As0G�   As0"P   As2�P   [4�As2��   As2�P   As5P   [4?�As5)�   As5P   As7uP   [49As7��   As7uP   As9�P   [2�As:�   As9�P   As<WP   [3��As<|�   As<WP   As>�P   [3�IAs>��   As>�P   AsA9P   [2��AsA^�   AsA9P   AsC�P   [2�AsC��   AsC�P   AsFP   [2�AsF@�   AsFP   AsH�P   [2w�AsH��   AsH�P   AsJ�P   [2�yAsK"�   AsJ�P   AsMnP   [1�AsM��   AsMnP   AsO�P   [1�3AsP�   AsO�P   AsRPP   [1�AsRu�   AsRPP   AsT�P   [1�qAsT��   AsT�P   AsW2P   [1��AsWW�   AsW2P   AsY�P   [1j�AsY��   AsY�P   As\P   [1��As\9�   As\P   As^�P   [0U�As^��   As^�P   As`�P   [0�}Asa�   As`�P   AscgP   [0u�Asc��   AscgP   Ase�P   [/��Ase��   Ase�P   AshIP   [0�aAshn�   AshIP   Asj�P   [0:�Asj��   Asj�P   Asm+P   [/�AsmP�   Asm+P   Aso�P   [0��Aso��   Aso�P   AsrP   [0��Asr2�   AsrP   Ast~P   [0V}Ast��   Ast~P   Asv�P   [0}�Asw�   Asv�P   Asy`P   [0Y�Asy��   Asy`P   As{�P   [/��As{��   As{�P   As~BP   [0$As~g�   As~BP   As��P   [0@rAs���   As��P   As�$P   [0fsAs�I�   As�$P   As��P   [0��As���   As��P   As�P   [/͇As�+�   As�P   As�wP   [/eAs���   As�wP   As��P   [/�6As��   As��P   As�YP   [.��As�~�   As�YP   As��P   [/��As���   As��P   As�;P   [.��As�`�   As�;P   As��P   [/&VAs���   As��P   As�P   [.�As�B�   As�P   As��P   [.�&As���   As��P   As��P   [0KAs�$�   As��P   As�pP   [/rAs���   As�pP   As��P   [09As��   As��P   As�RP   [/��As�w�   As�RP   As��P   [/i�As���   As��P   As�4P   [/��As�Y�   As�4P   As��P   [/�As���   As��P   As�P   [/9�As�;�   As�P   As��P   [.��As���   As��P   As��P   [.�wAs��   As��P   As�iP   [/�As���   As�iP   As��P   [.�(As���   As��P   As�KP   [.tzAs�p�   As�KP   As��P   [/�As���   As��P   As�-P   [0RAs�R�   As�-P   AsP   [/{aAs���   AsP   As�P   [/0:As�4�   As�P   AsǀP   [/k�Asǥ�   AsǀP   As��P   [/�As��   As��P   As�bP   [.qAṡ�   As�bP   As��P   [.n�As���   As��P   As�DP   [.G�As�i�   As�DP   AsӵP   [.�As���   AsӵP   As�&P   [.{�As�K�   As�&P   AsؗP   [.�;Asؼ�   AsؗP   As�P   [.�As�-�   As�P   As�yP   [.ӃAsݞ�   As�yP   As��P   [/As��   As��P   As�[P   [.��As��   As�[P   As��P   [/�AAs���   As��P   As�=P   [.}zAs�b�   As�=P   As�P   [/�As���   As�P   As�P   [.�As�D�   As�P   As�P   [.Z�As��   As�P   As�P   [/?�As�&�   As�P   As�rP   [/��As��   As�rP   As��P   [/3�As��   As��P   As�TP   [.�As�y�   As�TP   As��P   [.ټAs���   As��P   As�6P   [/�As�[�   As�6P   As��P   [.�As���   As��P   AtP   [/�At=�   AtP   At�P   [.ڊAt��   At�P   At�P   [.yFAt�   At�P   At	kP   [/whAt	��   At	kP   At�P   [/�9At�   At�P   AtMP   [.ȫAtr�   AtMP   At�P   [.��At��   At�P   At/P   [/�{AtT�   At/P   At�P   [/B)At��   At�P   AtP   [/�:At6�   AtP   At�P   [/|	At��   At�P   At�P   [/'�At�   At�P   AtdP   [/�>At��   AtdP   At!�P   [/KAt!��   At!�P   At$FP   [/M�At$k�   At$FP   At&�P   [.��At&��   At&�P   At)(P   [.�At)M�   At)(P   At+�P   [.ۅAt+��   At+�P   At.
P   [/�aAt./�   At.
P   At0{P   [/��At0��   At0{P   At2�P   [/�At3�   At2�P   At5]P   [.�At5��   At5]P   At7�P   [.GmAt7��   At7�P   At:?P   [/�TAt:d�   At:?P   At<�P   [/�JAt<��   At<�P   At?!P   [.�:At?F�   At?!P   AtA�P   [/e�AtA��   AtA�P   AtDP   [/9�AtD(�   AtDP   AtFtP   [.��AtF��   AtFtP   AtH�P   [/��AtI
�   AtH�P   AtKVP   [/�AtK{�   AtKVP   AtM�P   [0�`AtM��   AtM�P   AtP8P   [.�AtP]�   AtP8P   AtR�P   [/��AtR��   AtR�P   AtUP   [/SYAtU?�   AtUP   AtW�P   [/z�AtW��   AtW�P   AtY�P   [/4AtZ!�   AtY�P   At\mP   [/��At\��   At\mP   At^�P   [/;�At_�   At^�P   AtaOP   [/^	Atat�   AtaOP   Atc�P   [/��Atc��   Atc�P   Atf1P   [0�AtfV�   Atf1P   Ath�P   [/?iAth��   Ath�P   AtkP   [/�Atk8�   AtkP   Atm�P   [.��Atm��   Atm�P   Ato�P   [05Atp�   Ato�P   AtrfP   [.��Atr��   AtrfP   Att�P   [/�6Att��   Att�P   AtwHP   [/P�Atwm�   AtwHP   Aty�P   [/XRAty��   Aty�P   At|*P   [.At|O�   At|*P   At~�P   [/�At~��   At~�P   At�P   [06VAt�1�   At�P   At�}P   [/ЇAt���   At�}P   At��P   [/�<At��   At��P   At�_P   [0wlAt���   At�_P   At��P   [0c1At���   At��P   At�AP   [0O�At�f�   At�AP   At��P   [/A"At���   At��P   At�#P   [/ΕAt�H�   At�#P   At��P   [0!�At���   At��P   At�P   [/ʸAt�*�   At�P   At�vP   [/KAt���   At�vP   At��P   [/��At��   At��P   At�XP   [/�RAt�}�   At�XP   At��P   [1@At���   At��P   At�:P   [/V�At�_�   At�:P   At��P   [/�WAt���   At��P   At�P   [/��At�A�   At�P   At��P   [/��At���   At��P   At��P   [/�At�#�   At��P   At�oP   [0�At���   At�oP   At��P   [/��At��   At��P   At�QP   [/�At�v�   At�QP   At��P   [/�zAt���   At��P   At�3P   [/��At�X�   At�3P   At��P   [/�/At���   At��P   At�P   [/� At�:�   At�P   At��P   [0�At���   At��P   At��P   [/��At��   At��P   At�hP   [0�LAtō�   At�hP   At��P   [05At���   At��P   At�JP   [0��At�o�   At�JP   At̻P   [/T�At���   At̻P   At�,P   [/��At�Q�   At�,P   AtѝP   [/M�At���   AtѝP   At�P   [0�At�3�   At�P   At�P   [/r�At֤�   At�P   At��P   [.��At��   At��P   At�aP   [/4�Atۆ�   At�aP   At��P   [0�At���   At��P   At�CP   [/�lAt�h�   At�CP   At�P   [/�^At���   At�P   At�%P   [0 *At�J�   At�%P   At�P   [0NHAt��   At�P   At�P   [/ՎAt�,�   At�P   At�xP   [0/�At��   At�xP   At��P   [/��At��   At��P   At�ZP   [/��At��   At�ZP   At��P   [/}At���   At��P   At�<P   [/}gAt�a�   At�<P   At��P   [09�At���   At��P   At�P   [/BAt�C�   At�P   At��P   [/��At���   At��P   Au  P   [/C�Au %�   Au  P   AuqP   [0@`Au��   AuqP   Au�P   [/\Au�   Au�P   AuSP   [06Aux�   AuSP   Au	�P   [00#Au	��   Au	�P   Au5P   [/�AuZ�   Au5P   Au�P   [/ٚAu��   Au�P   AuP   [/)�Au<�   AuP   Au�P   [/ÊAu��   Au�P   Au�P   [/"UAu�   Au�P   AujP   [/+nAu��   AujP   Au�P   [0�Au �   Au�P   AuLP   [/QAuq�   AuLP   Au�P   [/�Au��   Au�P   Au".P   [0VlAu"S�   Au".P   Au$�P   [/Au$��   Au$�P   Au'P   [/}�Au'5�   Au'P   Au)�P   [/��Au)��   Au)�P   Au+�P   [/��Au,�   Au+�P   Au.cP   [/ �Au.��   Au.cP   Au0�P   [0�Au0��   Au0�P   Au3EP   [/N�Au3j�   Au3EP   Au5�P   [/1Au5��   Au5�P   Au8'P   [.�OAu8L�   Au8'P   Au:�P   [/6oAu:��   Au:�P   Au=	P   [/��Au=.�   Au=	P   Au?zP   [/��Au?��   Au?zP   AuA�P   [/��AuB�   AuA�P   AuD\P   [.�]AuD��   AuD\P   AuF�P   [/��AuF��   AuF�P   AuI>P   [/��AuIc�   AuI>P   AuK�P   [/�[AuK��   AuK�P   AuN P   [/��AuNE�   AuN P   AuP�P   [/>�AuP��   AuP�P   AuSP   [.�AuS'�   AuSP   AuUsP   [.�AuU��   AuUsP   AuW�P   [0^jAuX	�   AuW�P   AuZUP   [/�oAuZz�   AuZUP   Au\�P   [0x�Au\��   Au\�P   Au_7P   [.ԁAu_\�   Au_7P   Aua�P   [/�Aua��   Aua�P   AudP   [/�*Aud>�   AudP   Auf�P   [/�Auf��   Auf�P   Auh�P   [.��Aui �   Auh�P   AuklP   [.��Auk��   AuklP   Aum�P   [/?kAun�   Aum�P   AupNP   [/QAups�   AupNP   Aur�P   [/k�Aur��   Aur�P   Auu0P   [.�FAuuU�   Auu0P   Auw�P   [/�/Auw��   Auw�P   AuzP   [/�lAuz7�   AuzP   Au|�P   [/}PAu|��   Au|�P   Au~�P   [0%Au�   Au~�P   Au�eP   [/�5Au���   Au�eP   Au��P   [/��Au���   Au��P   Au�GP   [.��Au�l�   Au�GP   Au��P   [/:�Au���   Au��P   Au�)P   [.��Au�N�   Au�)P   Au��P   [/�Au���   Au��P   Au�P   [/|�Au�0�   Au�P   Au�|P   [/�oAu���   Au�|P   Au��P   [/�:Au��   Au��P   Au�^P   [/�#Au���   Au�^P   Au��P   [/�9Au���   Au��P   Au�@P   [/X�Au�e�   Au�@P   Au��P   [.�(Au���   Au��P   Au�"P   [/��Au�G�   Au�"P   Au��P   [.pgAu���   Au��P   Au�P   [/VAu�)�   Au�P   Au�uP   [/_Au���   Au�uP   Au��P   [/rWAu��   Au��P   Au�WP   [/��Au�|�   Au�WP   Au��P   [0X1Au���   Au��P   Au�9P   [/�3Au�^�   Au�9P   Au��P   [/I�Au���   Au��P   Au�P   [/��Au�@�   Au�P   Au��P   [/RAu���   Au��P   Au��P   [/4�Au�"�   Au��P   Au�nP   [/��Au���   Au�nP   Au��P   [/�Au��   Au��P   Au�PP   [/x�Au�u�   Au�PP   Au��P   [/dAu���   Au��P   Au�2P   [/�oAu�W�   Au�2P   AuʣP   [/B�Au���   AuʣP   Au�P   [/��Au�9�   Au�P   AuυP   [.w�AuϪ�   AuυP   Au��P   [/��Au��   Au��P   Au�gP   [/lyAuԌ�   Au�gP   Au��P   [/��Au���   Au��P   Au�IP   [.��Au�n�   Au�IP   AuۺP   [/m�Au���   AuۺP   Au�+P   [/5*Au�P�   Au�+P   Au��P   [.��Au���   Au��P   Au�P   [.�Au�2�   Au�P   Au�~P   [/��Au��   Au�~P   Au��P   [/��Au��   Au��P   Au�`P   [/�Au��   Au�`P   Au��P   [.�#Au���   Au��P   Au�BP   [.�/Au�g�   Au�BP   Au�P   [/o