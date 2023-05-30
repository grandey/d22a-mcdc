CDF   �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       oACCESS-ESM1.5 (2019): 
aerosol: CLASSIC (v1.0)
atmos: HadGAM2 (r1.1, N96; 192 x 145 longitude/latitude; 38 levels; top level 39255 m)
atmosChem: none
land: CABLE2.4
landIce: none
ocean: ACCESS-OM2 (MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: WOMBAT (same grid as ocean)
seaIce: CICE4.1 (same grid as ocean)    institution       aCommonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         @�f�       creation_date         2019-11-15T06:28:40Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Qhttps://furtherinfo.es-doc.org/CMIP6.CSIRO.ACCESS-ESM1-5.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (145x192 latxlon)   
grid_label        gn     history      �Tue May 30 16:58:32 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/ACCESS-ESM1-5_r1i1p1f1/rsut_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp245.r1i1p1f1.Amon.rsut.gn.v20191115/rsut_Amon_ACCESS-ESM1-5_ssp245_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp245_v20191115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/ACCESS-ESM1-5_r1i1p1f1/rsut_ACCESS-ESM1-5_r1i1p1f1_ssp245.mergetime.nc
Tue May 30 16:58:32 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rsut.gn.v20191115/rsut_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/ACCESS-ESM1-5_r1i1p1f1/rsut_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 05:53:37 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rsut.gn.v20191115/rsut_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.fx.areacella.gn.v20191115/areacella_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rsut.gn.v20191115/rsut_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191115.fldsum.nc
2019-11-15T06:28:40Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        CSIRO      mip_era       CMIP6      nominal_resolution        250 km     notes         FExp: ESM-historical; Local ID: HI-05; Variable: rsut (['fld_s01i208'])     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      ACCESS-ESM1-5      parent_time_units         days since 0101-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         ACCESS-ESM1-5      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         'ACCESS-ESM1-5 output prepared for CMIP6    variable_id       rsut   variant_label         r1i1p1f1   version       	v20191115      cmor_version      3.4.0      tracking_id       1hdl:21.14100/def487a6-592d-4143-930d-1805db1bda0e      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T                  	time_bnds                                    lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y                  rsut                   	   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   cell_measures         area: areacella    history       u2019-11-15T06:28:38Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               $                Aq���   Aq��P   Aq�P   [.؞Aq�6�   Aq�P   Aq��P   [.��Aq���   Aq��P   Aq��P   [.��Aq��   Aq��P   Aq�dP   [.��Aq���   Aq�dP   Aq��P   [.�&Aq���   Aq��P   Aq�FP   [.��Aq�k�   Aq�FP   Aq��P   [.�_Aq���   Aq��P   Aq�(P   [.�Aq�M�   Aq�(P   Aq��P   [/LZAq���   Aq��P   Aq�
P   [.RAq�/�   Aq�
P   Aq�{P   [.��Aq���   Aq�{P   Aq��P   [/��Aq��   Aq��P   Aq�]P   [/��AqĂ�   Aq�]P   Aq��P   [.��Aq���   Aq��P   Aq�?P   [/S�Aq�d�   Aq�?P   Aq˰P   [.�SAq���   Aq˰P   Aq�!P   [.Aq�F�   Aq�!P   AqВP   [.Aqз�   AqВP   Aq�P   [.�[Aq�(�   Aq�P   Aq�tP   [.��Aqՙ�   Aq�tP   Aq��P   [.-�Aq�
�   Aq��P   Aq�VP   [.:aAq�{�   Aq�VP   Aq��P   [.MAq���   Aq��P   Aq�8P   [.��Aq�]�   Aq�8P   Aq�P   [-�Aq���   Aq�P   Aq�P   [.�]Aq�?�   Aq�P   Aq�P   [.7�Aq��   Aq�P   Aq��P   [-��Aq�!�   Aq��P   Aq�mP   [.�OAq��   Aq�mP   Aq��P   [.��Aq��   Aq��P   Aq�OP   [.�dAq�t�   Aq�OP   Aq��P   [.�MAq���   Aq��P   Aq�1P   [.U�Aq�V�   Aq�1P   Aq��P   [0[�Aq���   Aq��P   Aq�P   [45�Aq�8�   Aq�P   Aq��P   [0�eAq���   Aq��P   Aq��P   [/YbAq��   Aq��P   ArfP   [/��Ar��   ArfP   Ar�P   [/k�Ar��   Ar�P   ArHP   [.,�Arm�   ArHP   Ar�P   [/e�Ar��   Ar�P   Ar*P   [/��ArO�   Ar*P   Ar�P   [/�LAr��   Ar�P   ArP   [.��Ar1�   ArP   Ar}P   [.��Ar��   Ar}P   Ar�P   [.RAr�   Ar�P   Ar_P   [/��Ar��   Ar_P   Ar�P   [.hfAr��   Ar�P   ArAP   [.y�Arf�   ArAP   Ar�P   [.��Ar��   Ar�P   Ar!#P   [.�rAr!H�   Ar!#P   Ar#�P   [/�Ar#��   Ar#�P   Ar&P   [/�Ar&*�   Ar&P   Ar(vP   [1E'Ar(��   Ar(vP   Ar*�P   [0pUAr+�   Ar*�P   Ar-XP   [/2*Ar-}�   Ar-XP   Ar/�P   [/�Ar/��   Ar/�P   Ar2:P   [/��Ar2_�   Ar2:P   Ar4�P   [/�Ar4��   Ar4�P   Ar7P   [/ZNAr7A�   Ar7P   Ar9�P   [-��Ar9��   Ar9�P   Ar;�P   [.�(Ar<#�   Ar;�P   Ar>oP   [1R�Ar>��   Ar>oP   Ar@�P   [0��ArA�   Ar@�P   ArCQP   [/+kArCv�   ArCQP   ArE�P   [.g`ArE��   ArE�P   ArH3P   [.��ArHX�   ArH3P   ArJ�P   [/��ArJ��   ArJ�P   ArMP   [/��ArM:�   ArMP   ArO�P   [0��ArO��   ArO�P   ArQ�P   [/�ArR�   ArQ�P   ArThP   [/�*ArT��   ArThP   ArV�P   [07iArV��   ArV�P   ArYJP   [.�pArYo�   ArYJP   Ar[�P   [/5bAr[��   Ar[�P   Ar^,P   [/BAr^Q�   Ar^,P   Ar`�P   [//{Ar`��   Ar`�P   ArcP   [.��Arc3�   ArcP   AreP   [0�Are��   AreP   Arg�P   [/_�Arh�   Arg�P   ArjaP   [/�Arj��   ArjaP   Arl�P   [/k�Arl��   Arl�P   AroCP   [/�SAroh�   AroCP   Arq�P   [0":Arq��   Arq�P   Art%P   [/CArtJ�   Art%P   Arv�P   [/�YArv��   Arv�P   AryP   [/�yAry,�   AryP   Ar{xP   [/>{Ar{��   Ar{xP   Ar}�P   [/&Ar~�   Ar}�P   Ar�ZP   [/r&Ar��   Ar�ZP   Ar��P   [/�Ar���   Ar��P   Ar�<P   [0%Ar�a�   Ar�<P   Ar��P   [/fZAr���   Ar��P   Ar�P   [/�IAr�C�   Ar�P   Ar��P   [/mpAr���   Ar��P   Ar� P   [/��Ar�%�   Ar� P   Ar�qP   [0x Ar���   Ar�qP   Ar��P   [/��Ar��   Ar��P   Ar�SP   [/��Ar�x�   Ar�SP   Ar��P   [/��Ar���   Ar��P   Ar�5P   [0 _Ar�Z�   Ar�5P   Ar��P   [/��Ar���   Ar��P   Ar�P   [/�Ar�<�   Ar�P   Ar��P   [/�Ar���   Ar��P   Ar��P   [/�Ar��   Ar��P   Ar�jP   [0c�Ar���   Ar�jP   Ar��P   [0�KAr� �   Ar��P   Ar�LP   [0<�Ar�q�   Ar�LP   Ar��P   [0?Ar���   Ar��P   Ar�.P   [0�sAr�S�   Ar�.P   Ar��P   [1-TAr���   Ar��P   Ar�P   [/�Ar�5�   Ar�P   Ar��P   [0I=Ar���   Ar��P   Ar��P   [3;Ar��   Ar��P   Ar�cP   [34�Ar���   Ar�cP   Ar��P   [1��Ar���   Ar��P   Ar�EP   [1'�Ar�j�   Ar�EP   ArĶP   [2��Ar���   ArĶP   Ar�'P   [0aAr�L�   Ar�'P   ArɘP   [0��Arɽ�   ArɘP   Ar�	P   [0�oAr�.�   Ar�	P   Ar�zP   [2]ArΟ�   Ar�zP   Ar��P   [1m�Ar��   Ar��P   Ar�\P   [0ΫArӁ�   Ar�\P   Ar��P   [1��Ar���   Ar��P   Ar�>P   [1x�Ar�c�   Ar�>P   ArگP   [1$iAr���   ArگP   Ar� P   [0�qAr�E�   Ar� P   ArߑP   [1[�Ar߶�   ArߑP   Ar�P   [0�&Ar�'�   Ar�P   Ar�sP   [0�Ar��   Ar�sP   Ar��P   [1w�Ar�	�   Ar��P   Ar�UP   [2}0Ar�z�   Ar�UP   Ar��P   [3<�Ar���   Ar��P   Ar�7P   [1��Ar�\�   Ar�7P   Ar�P   [2QMAr���   Ar�P   Ar�P   [0�Ar�>�   Ar�P   Ar��P   [1��Ar���   Ar��P   Ar��P   [0�CAr� �   Ar��P   Ar�lP   [1�>Ar���   Ar�lP   Ar��P   [1h1Ar��   Ar��P   Ar�NP   [2�Ar�s�   Ar�NP   As�P   [5�As��   As�P   As0P   [3H�AsU�   As0P   As�P   [1�bAs��   As�P   As	P   [/�As	7�   As	P   As�P   [/��As��   As�P   As�P   [/M.As�   As�P   AseP   [/e�As��   AseP   As�P   [/��As��   As�P   AsGP   [/�IAsl�   AsGP   As�P   [/rYAs��   As�P   As)P   [/��AsN�   As)P   As�P   [.��As��   As�P   AsP   [/'As0�   AsP   As!|P   [/As!��   As!|P   As#�P   [/��As$�   As#�P   As&^P   [/0�As&��   As&^P   As(�P   [.^�As(��   As(�P   As+@P   [.�As+e�   As+@P   As-�P   [.z�As-��   As-�P   As0"P   [/�As0G�   As0"P   As2�P   [.��As2��   As2�P   As5P   [.v�As5)�   As5P   As7uP   [.4KAs7��   As7uP   As9�P   [.*�As:�   As9�P   As<WP   [-��As<|�   As<WP   As>�P   [-�FAs>��   As>�P   AsA9P   [.5�AsA^�   AsA9P   AsC�P   [.-�AsC��   AsC�P   AsFP   [-�2AsF@�   AsFP   AsH�P   [-�~AsH��   AsH�P   AsJ�P   [-yeAsK"�   AsJ�P   AsMnP   [-XCAsM��   AsMnP   AsO�P   [-g�AsP�   AsO�P   AsRPP   [-5�AsRu�   AsRPP   AsT�P   [,��AsT��   AsT�P   AsW2P   [,-AsWW�   AsW2P   AsY�P   [,�kAsY��   AsY�P   As\P   [,�IAs\9�   As\P   As^�P   [-x]As^��   As^�P   As`�P   [,rAsa�   As`�P   AscgP   [+�JAsc��   AscgP   Ase�P   [,.qAse��   Ase�P   AshIP   [,!~Ashn�   AshIP   Asj�P   [,��Asj��   Asj�P   Asm+P   [,R�AsmP�   Asm+P   Aso�P   [,t�Aso��   Aso�P   AsrP   [+��Asr2�   AsrP   Ast~P   [+�Ast��   Ast~P   Asv�P   [+~Asw�   Asv�P   Asy`P   [+d�Asy��   Asy`P   As{�P   [+,As{��   As{�P   As~BP   [+�As~g�   As~BP   As��P   [+.	As���   As��P   As�$P   [+4As�I�   As�$P   As��P   [+J�As���   As��P   As�P   [+ �As�+�   As�P   As�wP   [+�As���   As�wP   As��P   [*�As��   As��P   As�YP   [+:�As�~�   As�YP   As��P   [+^#As���   As��P   As�;P   [*�As�`�   As�;P   As��P   [*hAs���   As��P   As�P   [*�As�B�   As�P   As��P   [*�-As���   As��P   As��P   [*�KAs�$�   As��P   As�pP   [*N�As���   As�pP   As��P   [* As��   As��P   As�RP   [*.�As�w�   As�RP   As��P   [++RAs���   As��P   As�4P   [*yZAs�Y�   As�4P   As��P   [*�As���   As��P   As�P   [)��As�;�   As�P   As��P   [*7AAs���   As��P   As��P   [)wIAs��   As��P   As�iP   [)��As���   As�iP   As��P   [*.qAs���   As��P   As�KP   [)Q�As�p�   As�KP   As��P   [*�As���   As��P   As�-P   [)+�As�R�   As�-P   AsP   [*V�As���   AsP   As�P   [*6�As�4�   As�P   AsǀP   [(�NAsǥ�   AsǀP   As��P   [)��As��   As��P   As�bP   [)�Aṡ�   As�bP   As��P   [)M8As���   As��P   As�DP   [(��As�i�   As�DP   AsӵP   [){DAs���   AsӵP   As�&P   [(��As�K�   As�&P   AsؗP   [)G�Asؼ�   AsؗP   As�P   [(��As�-�   As�P   As�yP   [)r�Asݞ�   As�yP   As��P   [(��As��   As��P   As�[P   [)"As��   As�[P   As��P   [(�qAs���   As��P   As�=P   [)(�As�b�   As�=P   As�P   [)As���   As�P   As�P   [(�bAs�D�   As�P   As�P   [(S�As��   As�P   As�P   [(��As�&�   As�P   As�rP   [(f�As��   As�rP   As��P   [(S�As��   As��P   As�TP   ['�ZAs�y�   As�TP   As��P   ['�uAs���   As��P   As�6P   ['��As�[�   As�6P   As��P   [(As���   As��P   AtP   [(iAt=�   AtP   At�P   [(U�At��   At�P   At�P   ['��At�   At�P   At	kP   [(�j