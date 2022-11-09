CDF   �   
      time       bnds      lon       lat          2   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �CMCC-ESM2 (2017): 
aerosol: MAM3
atmos: CAM5.3 (1deg; 288 x 192 longitude/latitude; 30 levels; top at ~2 hPa)
atmosChem: none
land: CLM4.5 (BGC mode)
landIce: none
ocean: NEMO3.6 (ORCA1 tripolar primarly 1 deg lat/lon with meridional refinement down to 1/3 degree in the tropics; 362 x 292 longitude/latitude; 50 vertical levels; top grid cell 0-1 m)
ocnBgchem: BFM5.1
seaIce: CICE4.0   institution       QFondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, Lecce 73100, Italy      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    comment       none   contact       	T. Lovato      creation_date         2020-12-21T16:03:04Z   data_specs_version        01.00.31   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.CMCC.CMCC-ESM2.historical.none.r1i1p1f1   grid      native atmosphere regular grid     
grid_label        gn     history      [Wed Nov 09 18:59:56 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/CMCC-ESM2_r1i1p1f1/rsut_CMCC-ESM2_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CMCC-ESM2_r1i1p1f1/CMIP6.ScenarioMIP.CMCC.CMCC-ESM2.ssp245.r1i1p1f1.Amon.rsut.gn.v20210129/rsut_Amon_CMCC-ESM2_ssp245_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp245_v20210129.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/CMCC-ESM2_r1i1p1f1/rsut_CMCC-ESM2_r1i1p1f1_ssp245.mergetime.nc
Wed Nov 09 18:59:55 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.Amon.rsut.gn.v20210114/rsut_Amon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20210114.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/CMCC-ESM2_r1i1p1f1/rsut_CMCC-ESM2_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 06:21:39 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.Amon.rsut.gn.v20210114/rsut_Amon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.fx.areacella.gn.v20210114/areacella_fx_CMCC-ESM2_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.Amon.rsut.gn.v20210114/rsut_Amon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20210114.fldsum.nc
2020-12-21T16:03:04Z ;rewrote data to be consistent with CMIP for variable rsut found in table Amon.;
none    initialization_index            institution_id        CMCC   mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      	CMCC-ESM2      parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      
references        none   run_variant       1st realization    	source_id         	CMCC-ESM2      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(05 February 2020) MD5:6a248fd76c55aa6d6f7b3cc6866b5faf      title         #CMCC-ESM2 output prepared for CMIP6    variable_id       rsut   variant_label         r1i1p1f1   license      ?CMIP6 model data produced by CMCC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.6.0      tracking_id       1hdl:21.14100/53052a79-7cd8-4193-a012-dc53a5c6aaa5      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   	   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   original_name         FSUTOA     cell_measures         area: areacella             �                Aq���   Aq��P   Aq�P   [.�$Aq�6�   Aq�P   Aq��P   [.��Aq���   Aq��P   Aq��P   [.�~Aq��   Aq��P   Aq�dP   [.kAq���   Aq�dP   Aq��P   [-�6Aq���   Aq��P   Aq�FP   [-��Aq�k�   Aq�FP   Aq��P   [.N�Aq���   Aq��P   Aq�(P   [.h.Aq�M�   Aq�(P   Aq��P   [.4�Aq���   Aq��P   Aq�
P   [.�Aq�/�   Aq�
P   Aq�{P   [/n�Aq���   Aq�{P   Aq��P   [/XZAq��   Aq��P   Aq�]P   [/�BAqĂ�   Aq�]P   Aq��P   [/y�Aq���   Aq��P   Aq�?P   [.(`Aq�d�   Aq�?P   Aq˰P   [-ѴAq���   Aq˰P   Aq�!P   [.�Aq�F�   Aq�!P   AqВP   [.�oAqз�   AqВP   Aq�P   [/��Aq�(�   Aq�P   Aq�tP   [.��Aqՙ�   Aq�tP   Aq��P   [/(Aq�
�   Aq��P   Aq�VP   [.��Aq�{�   Aq�VP   Aq��P   [.�)Aq���   Aq��P   Aq�8P   [.��Aq�]�   Aq�8P   Aq�P   [/p]Aq���   Aq�P   Aq�P   [.��Aq�?�   Aq�P   Aq�P   [.��Aq��   Aq�P   Aq��P   [.��Aq�!�   Aq��P   Aq�mP   [.��Aq��   Aq�mP   Aq��P   [/r�Aq��   Aq��P   Aq�OP   [0'5Aq�t�   Aq�OP   Aq��P   [/fxAq���   Aq��P   Aq�1P   [/"�Aq�V�   Aq�1P   Aq��P   [2p�Aq���   Aq��P   Aq�P   [6X^Aq�8�   Aq�P   Aq��P   [1Aq���   Aq��P   Aq��P   [/ڼAq��   Aq��P   ArfP   [/�LAr��   ArfP   Ar�P   [/�{Ar��   Ar�P   ArHP   [/�_Arm�   ArHP   Ar�P   [0�pAr��   Ar�P   Ar*P   [0'�ArO�   Ar*P   Ar�P   [0Ar��   Ar�P   ArP   [0DAr1�   ArP   Ar}P   [/nAr��   Ar}P   Ar�P   [/?TAr�   Ar�P   Ar_P   [/�Ar��   Ar_P   Ar�P   [/��Ar��   Ar�P   ArAP   [/�2Arf�   ArAP   Ar�P   [/@�Ar��   Ar�P   Ar!#P   [0-\Ar!H�   Ar!#P   Ar#�P   [/k�Ar#��   Ar#�P   Ar&P   [0٫Ar&*�   Ar&P   Ar(vP   [2��Ar(��   Ar(vP   Ar*�P   [0{Ar+�   Ar*�P   Ar-XP   [/6TAr-}�   Ar-XP   Ar/�P   [/B$Ar/��   Ar/�P   Ar2:P   [0�Ar2_�   Ar2:P   Ar4�P   [/Ar4��   Ar4�P   Ar7P   [/�Ar7A�   Ar7P   Ar9�P   [/�Ar9��   Ar9�P   Ar;�P   [/
�Ar<#�   Ar;�P   Ar>oP   [0�Ar>��   Ar>oP   Ar@�P   [1S+ArA�   Ar@�P   ArCQP   [/�ArCv�   ArCQP   ArE�P   [/*�ArE��   ArE�P   ArH3P   [.�<ArHX�   ArH3P   ArJ�P   [/&�ArJ��   ArJ�P   ArMP   [/ArM:�   ArMP   ArO�P   [.��ArO��   ArO�P   ArQ�P   [0�@ArR�   ArQ�P   ArThP   [/̊ArT��   ArThP   ArV�P   [/eKArV��   ArV�P   ArYJP   [/�ArYo�   ArYJP   Ar[�P   [.�Ar[��   Ar[�P   Ar^,P   [.�0Ar^Q�   Ar^,P   Ar`�P   [.�"Ar`��   Ar`�P   ArcP   [.ӫArc3�   ArcP   AreP   [/a6Are��   AreP   Arg�P   [.�Arh�   Arg�P   ArjaP   [/;�Arj��   ArjaP   Arl�P   [.nFArl��   Arl�P   AroCP   [/L�Aroh�   AroCP   Arq�P   [.��Arq��   Arq�P   Art%P   [.��ArtJ�   Art%P   Arv�P   [.y�Arv��   Arv�P   AryP   [.�kAry,�   AryP   Ar{xP   [/^%Ar{��   Ar{xP   Ar}�P   [.J�Ar~�   Ar}�P   Ar�ZP   [.��Ar��   Ar�ZP   Ar��P   [/��Ar���   Ar��P   Ar�<P   [/��Ar�a�   Ar�<P   Ar��P   [/dAr���   Ar��P   Ar�P   [.�-Ar�C�   Ar�P   Ar��P   [/kAr���   Ar��P   Ar� P   [/G�Ar�%�   Ar� P   Ar�qP   [.��Ar���   Ar�qP   Ar��P   [/�Ar��   Ar��P   Ar�SP   [/�Ar�x�   Ar�SP   Ar��P   [.�GAr���   Ar��P   Ar�5P   [0�0Ar�Z�   Ar�5P   Ar��P   [.�.Ar���   Ar��P   Ar�P   [/d�Ar�<�   Ar�P   Ar��P   [/��Ar���   Ar��P   Ar��P   [/E�Ar��   Ar��P   Ar�jP   [/lAr���   Ar�jP   Ar��P   [/4�Ar� �   Ar��P   Ar�LP   [/#�Ar�q�   Ar�LP   Ar��P   [/��Ar���   Ar��P   Ar�.P   [/K�Ar�S�   Ar�.P   Ar��P   [0L�Ar���   Ar��P   Ar�P   [/��Ar�5�   Ar�P   Ar��P   [0D�Ar���   Ar��P   Ar��P   [447Ar��   Ar��P   Ar�cP   [1��Ar���   Ar�cP   Ar��P   [0��Ar���   Ar��P   Ar�EP   [/(:Ar�j�   Ar�EP   ArĶP   [0Ar���   ArĶP   Ar�'P   [1�Ar�L�   Ar�'P   ArɘP   [/��Arɽ�   ArɘP   Ar�	P   [0��Ar�.�   Ar�	P   Ar�zP   [0��ArΟ�   Ar�zP   Ar��P   [0!�Ar��   Ar��P   Ar�\P   [/�ArӁ�   Ar�\P   Ar��P   [/��Ar���   Ar��P   Ar�>P   [/��Ar�c�   Ar�>P   ArگP   [/uAr���   ArگP   Ar� P   [/y�Ar�E�   Ar� P   ArߑP   [/�9Ar߶�   ArߑP   Ar�P   [/PeAr�'�   Ar�P   Ar�sP   [0��Ar��   Ar�sP   Ar��P   [/:tAr�	�   Ar��P   Ar�UP   [/�Ar�z�   Ar�UP   Ar��P   [0ʖAr���   Ar��P   Ar�7P   [0�3Ar�\�   Ar�7P   Ar�P   [/��Ar���   Ar�P   Ar�P   [/.%Ar�>�   Ar�P   Ar��P   [/��Ar���   Ar��P   Ar��P   [/QAr� �   Ar��P   Ar�lP   [/GrAr���   Ar�lP   Ar��P   [14�Ar��   Ar��P   Ar�NP   [1չAr�s�   Ar�NP   As�P   [5��As��   As�P   As0P   [2�VAsU�   As0P   As�P   [0=As��   As�P   As	P   [/uFAs	7�   As	P   As�P   [.�As��   As�P   As�P   [.0�As�   As�P   AseP   [/��As��   AseP   As�P   [.�As��   As�P   AsGP   [/��Asl�   AsGP   As�P   [-~rAs��   As�P   As)P   [/Y!AsN�   As)P   As�P   [/n�As��   As�P   AsP   [.�As0�   AsP   As!|P   [-�As!��   As!|P   As#�P   [.�As$�   As#�P   As&^P   [.9HAs&��   As&^P   As(�P   [.�As(��   As(�P   As+@P   [.^As+e�   As+@P   As-�P   [/G�As-��   As-�P   As0"P   [.�As0G�   As0"P   As2�P   [.�As2��   As2�P   As5P   [/j�As5)�   As5P   As7uP   [-�As7��   As7uP   As9�P   [/%As:�   As9�P   As<WP   [.��As<|�   As<WP   As>�P   [/J�As>��   As>�P   AsA9P   [0�#AsA^�   AsA9P   AsC�P   [/DFAsC��   AsC�P   AsFP   [0n�AsF@�   AsFP   AsH�P   [/,�AsH��   AsH�P   AsJ�P   [/K�AsK"�   AsJ�P   AsMnP   [/7�AsM��   AsMnP   AsO�P   [.~'AsP�   AsO�P   AsRPP   [.2�AsRu�   AsRPP   AsT�P   [.�@AsT��   AsT�P   AsW2P   [/}�AsWW�   AsW2P   AsY�P   [.seAsY��   AsY�P   As\P   [-��As\9�   As\P   As^�P   [/�As^��   As^�P   As`�P   [-�1Asa�   As`�P   AscgP   [.A�Asc��   AscgP   Ase�P   [-�JAse��   Ase�P   AshIP   [-��Ashn�   AshIP   Asj�P   [-\zAsj��   Asj�P   Asm+P   [,��AsmP�   Asm+P   Aso�P   [-BAAso��   Aso�P   AsrP   [,��Asr2�   AsrP   Ast~P   [,��Ast��   Ast~P   Asv�P   [-��Asw�   Asv�P   Asy`P   [,�Asy��   Asy`P   As{�P   [,r;As{��   As{�P   As~BP   [+�As~g�   As~BP   As��P   [+QAAs���   As��P   As�$P   [+�\As�I�   As�$P   As��P   [+,9As���   As��P   As�P   [+w�As�+�   As�P   As�wP   [+yNAs���   As�wP   As��P   [+D�As��   As��P   As�YP   [,D�As�~�   As�YP   As��P   [*oAs���   As��P   As�;P   [*'�As�`�   As�;P   As��P   [*�As���   As��P   As�P   [*�TAs�B�   As�P   As��P   [*|�As���   As��P   As��P   [)�As�$�   As��P   As�pP   [)ٷAs���   As�pP   As��P   [*-As��   As��P   As�RP   [)��As�w�   As�RP   As��P   [+m�As���   As��P   As�4P   [)CAs�Y�   As�4P   As��P   [(��As���   As��P   As�P   [)�hAs�;�   As�P   As��P   [)��As���   As��P   As��P   [)�`As��   As��P   As�iP   [)�As���   As�iP   As��P   [)0?As���   As��P   As�KP   [(��As�p�   As�KP   As��P   [)�As���   As��P   As�-P   [)R�As�R�   As�-P   AsP   [)UAs���   AsP   As�P   [(�RAs�4�   As�P   AsǀP   [(��Asǥ�   AsǀP   As��P   [(j1As��   As��P   As�bP   [(�cAṡ�   As�bP   As��P   [(N_As���   As��P   As�DP   [(�
As�i�   As�DP   AsӵP   [(��As���   AsӵP   As�&P   ['�bAs�K�   As�&P   AsؗP   [)#�Asؼ�   AsؗP   As�P   [(��As�-�   As�P   As�yP   [(n#Asݞ�   As�yP   As��P   [(H�As��   As��P   As�[P   ['�8As��   As�[P   As��P   ['��As���   As��P   As�=P   ['��As�b�   As�=P   As�P   ['ƢAs���   As�P   As�P   [(Z'As�D�   As�P   As�P   ['��As��   As�P   As�P   [(�BAs�&�   As�P   As�rP   ['[�As��   As�rP   As��P   ['<As��   As��P   As�TP   ['��As�y�   As�TP   As��P   ['q	As���   As��P   As�6P   ['a�As�[�   As�6P   As��P   ['!�As���   As��P   AtP   ['0�At=�   AtP   At�P   ['�tAt��   At�P   At�P   [&�At�   At�P   At	kP   ['�