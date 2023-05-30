CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       oACCESS-ESM1.5 (2019): 
aerosol: CLASSIC (v1.0)
atmos: HadGAM2 (r1.1, N96; 192 x 145 longitude/latitude; 38 levels; top level 39255 m)
atmosChem: none
land: CABLE2.4
landIce: none
ocean: ACCESS-OM2 (MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: WOMBAT (same grid as ocean)
seaIce: CICE4.1 (same grid as ocean)    institution       aCommonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         @�f�       creation_date         2019-11-15T06:28:14Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Qhttps://furtherinfo.es-doc.org/CMIP6.CSIRO.ACCESS-ESM1-5.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (145x192 latxlon)   
grid_label        gn     history      �Tue May 30 16:58:16 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/ACCESS-ESM1-5_r1i1p1f1/rsdt_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsdt.gn.v20210318/rsdt_Amon_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp126_v20210318.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsdt.gn.v20210318/rsdt_Amon_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_210101-230012.yearmean.mul.areacella_ssp126_v20210318.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/ACCESS-ESM1-5_r1i1p1f1/rsdt_ACCESS-ESM1-5_r1i1p1f1_ssp126.mergetime.nc
Tue May 30 16:58:15 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rsdt.gn.v20191115/rsdt_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/ACCESS-ESM1-5_r1i1p1f1/rsdt_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 04:13:10 2022: cdo -O -s -fldsum -setattribute,rsdt@units=W m-2 m2 -mul -yearmean -selname,rsdt /Users/benjamin/Data/p22b/CMIP6/rsdt/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rsdt.gn.v20191115/rsdt_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.fx.areacella.gn.v20191115/areacella_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rsdt.gn.v20191115/rsdt_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191115.fldsum.nc
2019-11-15T06:28:14Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.    initialization_index            institution_id        CSIRO      mip_era       CMIP6      nominal_resolution        250 km     notes         FExp: ESM-historical; Local ID: HI-05; Variable: rsdt (['fld_s01i207'])     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      ACCESS-ESM1-5      parent_time_units         days since 0101-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         ACCESS-ESM1-5      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         'ACCESS-ESM1-5 output prepared for CMIP6    variable_id       rsdt   variant_label         r1i1p1f1   version       	v20191115      cmor_version      3.4.0      tracking_id       1hdl:21.14100/09deaf5e-87ef-4a52-9aea-b948c014404a      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               4   	time_bnds                                 <   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               $   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               ,   rsdt                   	   standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       9Shortwave radiation incident at the top of the atmosphere      cell_measures         area: areacella    history       u2019-11-15T06:28:12Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               L                Aq���   Aq��P   Aq�P   \��Aq�6�   Aq�P   Aq��P   \�NAq���   Aq��P   Aq��P   \�'Aq��   Aq��P   Aq�dP   \��Aq���   Aq�dP   Aq��P   \��Aq���   Aq��P   Aq�FP   \��Aq�k�   Aq�FP   Aq��P   \�3Aq���   Aq��P   Aq�(P   \��Aq�M�   Aq�(P   Aq��P   \�uAq���   Aq��P   Aq�
P   \��Aq�/�   Aq�
P   Aq�{P   \�Aq���   Aq�{P   Aq��P   \��Aq��   Aq��P   Aq�]P   \��AqĂ�   Aq�]P   Aq��P   \�2Aq���   Aq��P   Aq�?P   \��Aq�d�   Aq�?P   Aq˰P   \�NAq���   Aq˰P   Aq�!P   \��Aq�F�   Aq�!P   AqВP   \��Aqз�   AqВP   Aq�P   \��Aq�(�   Aq�P   Aq�tP   \��Aqՙ�   Aq�tP   Aq��P   \�3Aq�
�   Aq��P   Aq�VP   \��Aq�{�   Aq�VP   Aq��P   \�LAq���   Aq��P   Aq�8P   \�,Aq�]�   Aq�8P   Aq�P   \�JAq���   Aq�P   Aq�P   \��Aq�?�   Aq�P   Aq�P   \�Aq��   Aq�P   Aq��P   \�Aq�!�   Aq��P   Aq�mP   \��Aq��   Aq�mP   Aq��P   \�
Aq��   Aq��P   Aq�OP   \�/Aq�t�   Aq�OP   Aq��P   \��Aq���   Aq��P   Aq�1P   \��Aq�V�   Aq�1P   Aq��P   \��Aq���   Aq��P   Aq�P   \�>Aq�8�   Aq�P   Aq��P   \��Aq���   Aq��P   Aq��P   \�IAq��   Aq��P   ArfP   \��Ar��   ArfP   Ar�P   \�Ar��   Ar�P   ArHP   \��Arm�   ArHP   Ar�P   \��Ar��   Ar�P   Ar*P   \�DArO�   Ar*P   Ar�P   \��Ar��   Ar�P   ArP   \��Ar1�   ArP   Ar}P   \�HAr��   Ar}P   Ar�P   \�
Ar�   Ar�P   Ar_P   \�Ar��   Ar_P   Ar�P   \�wAr��   Ar�P   ArAP   \��Arf�   ArAP   Ar�P   \��Ar��   Ar�P   Ar!#P   \�kAr!H�   Ar!#P   Ar#�P   \�\Ar#��   Ar#�P   Ar&P   \��Ar&*�   Ar&P   Ar(vP   \�Ar(��   Ar(vP   Ar*�P   \��Ar+�   Ar*�P   Ar-XP   \�0Ar-}�   Ar-XP   Ar/�P   \��Ar/��   Ar/�P   Ar2:P   \��Ar2_�   Ar2:P   Ar4�P   \��Ar4��   Ar4�P   Ar7P   \��Ar7A�   Ar7P   Ar9�P   \��Ar9��   Ar9�P   Ar;�P   \��Ar<#�   Ar;�P   Ar>oP   \�AAr>��   Ar>oP   Ar@�P   \��ArA�   Ar@�P   ArCQP   \��ArCv�   ArCQP   ArE�P   \��ArE��   ArE�P   ArH3P   \�ArHX�   ArH3P   ArJ�P   \��ArJ��   ArJ�P   ArMP   \��ArM:�   ArMP   ArO�P   \�<ArO��   ArO�P   ArQ�P   \�AArR�   ArQ�P   ArThP   \��ArT��   ArThP   ArV�P   \��ArV��   ArV�P   ArYJP   \�ArYo�   ArYJP   Ar[�P   \��Ar[��   Ar[�P   Ar^,P   \��Ar^Q�   Ar^,P   Ar`�P   \�Ar`��   Ar`�P   ArcP   \�cArc3�   ArcP   AreP   \��Are��   AreP   Arg�P   \�Arh�   Arg�P   ArjaP   \�&Arj��   ArjaP   Arl�P   \�Arl��   Arl�P   AroCP   \�EAroh�   AroCP   Arq�P   \��Arq��   Arq�P   Art%P   \�yArtJ�   Art%P   Arv�P   \�~Arv��   Arv�P   AryP   \ƏAry,�   AryP   Ar{xP   \�`Ar{��   Ar{xP   Ar}�P   \��Ar~�   Ar}�P   Ar�ZP   \�UAr��   Ar�ZP   Ar��P   \��Ar���   Ar��P   Ar�<P   \�UAr�a�   Ar�<P   Ar��P   \�Ar���   Ar��P   Ar�P   \��Ar�C�   Ar�P   Ar��P   \�Ar���   Ar��P   Ar� P   \�Ar�%�   Ar� P   Ar�qP   \��Ar���   Ar�qP   Ar��P   \�EAr��   Ar��P   Ar�SP   \��Ar�x�   Ar�SP   Ar��P   \�PAr���   Ar��P   Ar�5P   \��Ar�Z�   Ar�5P   Ar��P   \��Ar���   Ar��P   Ar�P   \�<Ar�<�   Ar�P   Ar��P   \�NAr���   Ar��P   Ar��P   \�TAr��   Ar��P   Ar�jP   \��Ar���   Ar�jP   Ar��P   \�Ar� �   Ar��P   Ar�LP   \��Ar�q�   Ar�LP   Ar��P   \׏Ar���   Ar��P   Ar�.P   \�sAr�S�   Ar�.P   Ar��P   \�hAr���   Ar��P   Ar�P   \��Ar�5�   Ar�P   Ar��P   \��Ar���   Ar��P   Ar��P   \�PAr��   Ar��P   Ar�cP   \�Ar���   Ar�cP   Ar��P   \��Ar���   Ar��P   Ar�EP   \��Ar�j�   Ar�EP   ArĶP   \��Ar���   ArĶP   Ar�'P   \ǖAr�L�   Ar�'P   ArɘP   \�AArɽ�   ArɘP   Ar�	P   \ǣAr�.�   Ar�	P   Ar�zP   \�;ArΟ�   Ar�zP   Ar��P   \��Ar��   Ar��P   Ar�\P   \�AArӁ�   Ar�\P   Ar��P   \�Ar���   Ar��P   Ar�>P   \� Ar�c�   Ar�>P   ArگP   \�Ar���   ArگP   Ar� P   \��Ar�E�   Ar� P   ArߑP   \�LAr߶�   ArߑP   Ar�P   \�$Ar�'�   Ar�P   Ar�sP   \�SAr��   Ar�sP   Ar��P   \��Ar�	�   Ar��P   Ar�UP   \�Ar�z�   Ar�UP   Ar��P   \��Ar���   Ar��P   Ar�7P   \�"Ar�\�   Ar�7P   Ar�P   \�LAr���   Ar�P   Ar�P   \��Ar�>�   Ar�P   Ar��P   \�7Ar���   Ar��P   Ar��P   \�qAr� �   Ar��P   Ar�lP   \�jAr���   Ar�lP   Ar��P   \��Ar��   Ar��P   Ar�NP   \̓Ar�s�   Ar�NP   As�P   \��As��   As�P   As0P   \��AsU�   As0P   As�P   \�
As��   As�P   As	P   \��As	7�   As	P   As�P   \��As��   As�P   As�P   \��As�   As�P   AseP   \�DAs��   AseP   As�P   \�As��   As�P   AsGP   \��Asl�   AsGP   As�P   \�pAs��   As�P   As)P   \ͺAsN�   As)P   As�P   \��As��   As�P   AsP   \��As0�   AsP   As!|P   \��As!��   As!|P   As#�P   \�JAs$�   As#�P   As&^P   \��As&��   As&^P   As(�P   \�As(��   As(�P   As+@P   \�As+e�   As+@P   As-�P   \�#As-��   As-�P   As0"P   \��As0G�   As0"P   As2�P   \�,As2��   As2�P   As5P   \�As5)�   As5P   As7uP   \�:As7��   As7uP   As9�P   \��As:�   As9�P   As<WP   \�(As<|�   As<WP   As>�P   \��As>��   As>�P   AsA9P   \�AsA^�   AsA9P   AsC�P   \��AsC��   AsC�P   AsFP   \�AsF@�   AsFP   AsH�P   \�HAsH��   AsH�P   AsJ�P   \�FAsK"�   AsJ�P   AsMnP   \�pAsM��   AsMnP   AsO�P   \ˀAsP�   AsO�P   AsRPP   \ŃAsRu�   AsRPP   AsT�P   \��AsT��   AsT�P   AsW2P   \�AsWW�   AsW2P   AsY�P   \��AsY��   AsY�P   As\P   \��As\9�   As\P   As^�P   \��As^��   As^�P   As`�P   \��Asa�   As`�P   AscgP   \�:Asc��   AscgP   Ase�P   \�1Ase��   Ase�P   AshIP   \��Ashn�   AshIP   Asj�P   \��Asj��   Asj�P   Asm+P   \��AsmP�   Asm+P   Aso�P   \��Aso��   Aso�P   AsrP   \��Asr2�   AsrP   Ast~P   \�3Ast��   Ast~P   Asv�P   \��Asw�   Asv�P   Asy`P   \�XAsy��   Asy`P   As{�P   \�As{��   As{�P   As~BP   \��As~g�   As~BP   As��P   \�wAs���   As��P   As�$P   \��As�I�   As�$P   As��P   \�AAs���   As��P   As�P   \�uAs�+�   As�P   As�wP   \�$As���   As�wP   As��P   \��As��   As��P   As�YP   \�@As�~�   As�YP   As��P   \�_As���   As��P   As�;P   \��As�`�   As�;P   As��P   \�?As���   As��P   As�P   \�9As�B�   As�P   As��P   \��As���   As��P   As��P   \��As�$�   As��P   As�pP   \�As���   As�pP   As��P   \�As��   As��P   As�RP   \��As�w�   As�RP   As��P   \��As���   As��P   As�4P   \��As�Y�   As�4P   As��P   \�FAs���   As��P   As�P   \�As�;�   As�P   As��P   \��As���   As��P   As��P   \�As��   As��P   As�iP   \��As���   As�iP   As��P   \�RAs���   As��P   As�KP   \��As�p�   As�KP   As��P   \�JAs���   As��P   As�-P   \��As�R�   As�-P   AsP   \��As���   AsP   As�P   \�As�4�   As�P   AsǀP   \�IAsǥ�   AsǀP   As��P   \��As��   As��P   As�bP   \��Aṡ�   As�bP   As��P   \�`As���   As��P   As�DP   \��As�i�   As�DP   AsӵP   \��As���   AsӵP   As�&P   \�WAs�K�   As�&P   AsؗP   \�iAsؼ�   AsؗP   As�P   \��As�-�   As�P   As�yP   \��Asݞ�   As�yP   As��P   \�PAs��   As��P   As�[P   \��As��   As�[P   As��P   \��As���   As��P   As�=P   \�$As�b�   As�=P   As�P   \��As���   As�P   As�P   \�FAs�D�   As�P   As�P   \�As��   As�P   As�P   \�aAs�&�   As�P   As�rP   \��As��   As�rP   As��P   \��As��   As��P   As�TP   \��As�y�   As�TP   As��P   \��As���   As��P   As�6P   \��As�[�   As�6P   As��P   \��As���   As��P   AtP   \�3At=�   AtP   At�P   \��At��   At�P   At�P   \��At�   At�P   At	kP   \��At	��   At	kP   At�P   \��At�   At�P   AtMP   \�lAtr�   AtMP   At�P   \�At��   At�P   At/P   \��AtT�   At/P   At�P   \��At��   At�P   AtP   \�At6�   AtP   At�P   \��At��   At�P   At�P   \��At�   At�P   AtdP   \�At��   AtdP   At!�P   \��At!��   At!�P   At$FP   \��At$k�   At$FP   At&�P   \��At&��   At&�P   At)(P   \�~At)M�   At)(P   At+�P   \��At+��   At+�P   At.
P   \�>At./�   At.
P   At0{P   \�bAt0��   At0{P   At2�P   \��At3�   At2�P   At5]P   \�MAt5��   At5]P   At7�P   \�TAt7��   At7�P   At:?P   \��At:d�   At:?P   At<�P   \��At<��   At<�P   At?!P   \�,At?F�   At?!P   AtA�P   \��AtA��   AtA�P   AtDP   \��AtD(�   AtDP   AtFtP   \��AtF��   AtFtP   AtH�P   \�rAtI
�   AtH�P   AtKVP   \��AtK{�   AtKVP   AtM�P   \��AtM��   AtM�P   AtP8P   \�6AtP]�   AtP8P   AtR�P   \��AtR��   AtR�P   AtUP   \�`AtU?�   AtUP   AtW�P   \�:AtW��   AtW�P   AtY�P   \ÍAtZ!�   AtY�P   At\mP   \��At\��   At\mP   At^�P   \��At_�   At^�P   AtaOP   \��Atat�   AtaOP   Atc�P   \��Atc��   Atc�P   Atf1P   \��AtfV�   Atf1P   Ath�P   \��Ath��   Ath�P   AtkP   \�kAtk8�   AtkP   Atm�P   \�Atm��   Atm�P   Ato�P   \��Atp�   Ato�P   AtrfP   \��Atr��   AtrfP   Att�P   \��Att��   Att�P   AtwHP   \��Atwm�   AtwHP   Aty�P   \��Aty��   Aty�P   At|*P   \�2At|O�   At|*P   At~�P   \�aAt~��   At~�P   At�P   \��At�1�   At�P   At�}P   \�.At���   At�}P   At��P   \�HAt��   At��P   At�_P   \�]At���   At�_P   At��P   \�MAt���   At��P   At�AP   \�kAt�f�   At�AP   At��P   \�!At���   At��P   At�#P   \��At�H�   At�#P   At��P   \�RAt���   At��P   At�P   \��At�*�   At�P   At�vP   \�mAt���   At�vP   At��P   \��At��   At��P   At�XP   \�IAt�}�   At�XP   At��P   \�VAt���   At��P   At�:P   \��At�_�   At�:P   At��P   \�At���   At��P   At�P   \�At�A�   At�P   At��P   \�"At���   At��P   At��P   \��At�#�   At��P   At�oP   \�At���   At�oP   At��P   \�(At��   At��P   At�QP   \��At�v�   At�QP   At��P   \�^At���   At��P   At�3P   \��At�X�   At�3P   At��P   \��At���   At��P   At�P   \��At�:�   At�P   At��P   \�uAt���   At��P   At��P   \��At��   At��P   At�hP   \�Atō�   At�hP   At��P   \��At���   At��P   At�JP   \��At�o�   At�JP   At̻P   \��At���   At̻P   At�,P   \��At�Q�   At�,P   AtѝP   \�At���   AtѝP   At�P   \�GAt�3�   At�P   At�P   \�At֤�   At�P   At��P   \�PAt��   At��P   At�aP   \��Atۆ�   At�aP   At��P   \��At���   At��P   At�CP   \�/At�h�   At�CP   At�P   \��At���   At�P   At�%P   \��At�J�   At�%P   At�P   \�At��   At�P   At�P   \�At�,�   At�P   At�xP   \�'At��   At�xP   At��P   \��At��   At��P   At�ZP   \��At��   At�ZP   At��P   \�LAt���   At��P   At�<P   \�bAt�a�   At�<P   At��P   \�mAt���   At��P   At�P   \��At�C�   At�P   At��P   \�qAt���   At��P   Au  P   \��Au %�   Au  P   AuqP   \�BAu��   AuqP   Au�P   \�cAu�   Au�P   AuSP   \�8Aux�   AuSP   Au	�P   \�Au	��   Au	�P   Au5P   \��AuZ�   Au5P   Au�P   \�Au��   Au�P   AuP   \�AAu<�   AuP   Au�P   \��Au��   Au�P   Au�P   \�gAu�   Au�P   AujP   \�QAu��   AujP   Au�P   \��Au �   Au�P   AuLP   \�|Auq�   AuLP   Au�P   \�xAu��   Au�P   Au".P   \�BAu"S�   Au".P   Au$�P   \�DAu$��   Au$�P   Au'P   \�Au'5�   Au'P   Au)�P   \��Au)��   Au)�P   Au+�P   \��Au,�   Au+�P   Au.cP   \��Au.��   Au.cP   Au0�P   \��Au0��   Au0�P   Au3EP   \�XAu3j�   Au3EP   Au5�P   \�uAu5��   Au5�P   Au8'P   \�[Au8L�   Au8'P   Au:�P   \�Au:��   Au:�P   Au=	P   \��Au=.�   Au=	P   Au?zP   \�;Au?��   Au?zP   AuA�P   \��AuB�   AuA�P   AuD\P   \�aAuD��   AuD\P   AuF�P   \�AuF��   AuF�P   AuI>P   \��AuIc�   AuI>P   AuK�P   \��AuK��   AuK�P   AuN P   \��AuNE�   AuN P   AuP�P   \�LAuP��   AuP�P   AuSP   \��AuS'�   AuSP   AuUsP   \��AuU��   AuUsP   AuW�P   \�hAuX	�   AuW�P   AuZUP   \�AuZz�   AuZUP   Au\�P   \�ZAu\��   Au\�P   Au_7P   \�.Au_\�   Au_7P   Aua�P   \�9Aua��   Aua�P   AudP   \��Aud>�   AudP   Auf�P   \�4Auf��   Auf�P   Auh�P   \�fAui �   Auh�P   AuklP   \��Auk��   AuklP   Aum�P   \�Aun�   Aum�P   AupNP   \��Aups�   AupNP   Aur�P   \��Aur��   Aur�P   Auu0P   \��AuuU�   Auu0P   Auw�P   \�>Auw��   Auw�P   AuzP   \�Auz7�   AuzP   Au|�P   \�CAu|��   Au|�P   Au~�P   \��Au�   Au~�P   Au�eP   \�_Au���   Au�eP   Au��P   \�6Au���   Au��P   Au�GP   \��Au�l�   Au�GP   Au��P   \��Au���   Au��P   Au�)P   \�	Au�N�   Au�)P   Au��P   \��Au���   Au��P   Au�P   \��Au�0�   Au�P   Au�|P   \�xAu���   Au�|P   Au��P   \��Au��   Au��P   Au�^P   \��Au���   Au�^P   Au��P   \�RAu���   Au��P   Au�@P   \�"Au�e�   Au�@P   Au��P   \�DAu���   Au��P   Au�"P   \��Au�G�   Au�"P   Au��P   \��Au���   Au��P   Au�P   \��Au�)�   Au�P   Au�uP   \�vAu���   Au�uP   Au��P   \�=Au��   Au��P   Au�WP   \�Au�|�   Au�WP   Au��P   \�Au���   Au��P   Au�9P   \��Au�^�   Au�9P   Au��P   \�!Au���   Au��P   Au�P   \��Au�@�   Au�P   Au��P   \��Au���   Au��P   Au��P   \�&Au�"�   Au��P   Au�nP   \�0Au���   Au�nP   Au��P   \��Au��   Au��P   Au�PP   \�;Au�u�   Au�PP   Au��P   \�.Au���   Au��P   Au�2P   \��Au�W�   Au�2P   AuʣP   \�7Au���   AuʣP   Au�P   \��Au�9�   Au�P   AuυP   \��AuϪ�   AuυP   Au��P   \��Au��   Au��P   Au�gP   \��AuԌ�   Au�gP   Au��P   \��Au���   Au��P   Au�IP   \��Au�n�   Au�IP   AuۺP   \��Au���   AuۺP   Au�+P   \�iAu�P�   Au�+P   Au��P   \�}Au���   Au��P   Au�P   \��Au�2�   Au�P   Au�~P   \��Au��   Au�~P   Au��P   \�)Au��   Au��P   Au�`P   \��Au��   Au�`P   Au��P   \�Au���   Au��P   Au�BP   \��Au�g�   Au�BP   Au�P   \�V