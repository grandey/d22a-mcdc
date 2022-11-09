CDF  �   
      time       bnds      lon       lat          7   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-04-23T09:02:21Z   description       DECK: control      title         <CNRM-ESM2-1 model output prepared for CMIP6 / CMIP piControl   activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.1    experiment_id         	piControl      
experiment        pre-industrial control     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.piControl.none.r1i1p1f2      grid      ldata regridded to a T127 gaussian grid (128x256 latlon) from a native atmosphere T127l reduced gaussian grid   
grid_label        gr     nominal_resolution        250 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent         @�Հ       branch_time_in_child                 physics_index               product       model-output   realization_index               realm         atmos      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Amon   variable_id       rsdt   variant_info      �. Information provided by this attribute may in some cases be flawed. Users can find more comprehensive and up-to-date documentation via the further_info_url global attribute.    variant_label         r1i1p1f2   EXPID         CNRM-ESM2-1_piControl_r1i1p1f2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          87374385b726e2a5f1e17b33af88ce8c   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.1      history      �Wed Nov 09 18:59:13 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Amon.rsdt.gr.v20181115/rsdt_Amon_CNRM-ESM2-1_piControl_r1i1p1f2_gr_185001-234912.yearmean.mul.areacella_piControl_v20181115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/CNRM-ESM2-1_r1i1p1f2/rsdt_CNRM-ESM2-1_r1i1p1f2_piControl.mergetime.nc
Fri Nov 04 04:42:05 2022: cdo -O -s -fldsum -setattribute,rsdt@units=W m-2 m2 -mul -yearmean -selname,rsdt /Users/benjamin/Data/p22b/CMIP6/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Amon.rsdt.gr.v20181115/rsdt_Amon_CNRM-ESM2-1_piControl_r1i1p1f2_gr_185001-234912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.fx.areacella.gr.v20181115/areacella_fx_CNRM-ESM2-1_piControl_r1i1p1f2_gr.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Amon.rsdt.gr.v20181115/rsdt_Amon_CNRM-ESM2-1_piControl_r1i1p1f2_gr_185001-234912.yearmean.mul.areacella_piControl_v20181115.fldsum.nc
none      tracking_id       1hdl:21.14100/c0c28c93-a2bb-47d0-a9d9-becde8485aca      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T                   	time_bnds                                    lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsdt                      standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   online_operation      average    interval_operation        900 s      interval_write        1 month    description       at the top of the atmosphere   positive      down   history       none   cell_measures         area: areacella                             Aq���   Aq��P   Aq�P   \@�Aq�6�   Aq�P   Aq��P   \@�Aq���   Aq��P   Aq��P   \A�Aq��   Aq��P   Aq�dP   \@�Aq���   Aq�dP   Aq��P   \@�Aq���   Aq��P   Aq�FP   \@�Aq�k�   Aq�FP   Aq��P   \A�Aq���   Aq��P   Aq�(P   \@�Aq�M�   Aq�(P   Aq��P   \@�Aq���   Aq��P   Aq�
P   \@�Aq�/�   Aq�
P   Aq�{P   \A�Aq���   Aq�{P   Aq��P   \@�Aq��   Aq��P   Aq�]P   \@�AqĂ�   Aq�]P   Aq��P   \@�Aq���   Aq��P   Aq�?P   \A�Aq�d�   Aq�?P   Aq˰P   \@�Aq���   Aq˰P   Aq�!P   \@�Aq�F�   Aq�!P   AqВP   \@�Aqз�   AqВP   Aq�P   \A�Aq�(�   Aq�P   Aq�tP   \@�Aqՙ�   Aq�tP   Aq��P   \@�Aq�
�   Aq��P   Aq�VP   \@�Aq�{�   Aq�VP   Aq��P   \A�Aq���   Aq��P   Aq�8P   \@�Aq�]�   Aq�8P   Aq�P   \@�Aq���   Aq�P   Aq�P   \@�Aq�?�   Aq�P   Aq�P   \A�Aq��   Aq�P   Aq��P   \@�Aq�!�   Aq��P   Aq�mP   \@�Aq��   Aq�mP   Aq��P   \@�Aq��   Aq��P   Aq�OP   \A�Aq�t�   Aq�OP   Aq��P   \@�Aq���   Aq��P   Aq�1P   \@�Aq�V�   Aq�1P   Aq��P   \@�Aq���   Aq��P   Aq�P   \A�Aq�8�   Aq�P   Aq��P   \@�Aq���   Aq��P   Aq��P   \@�Aq��   Aq��P   ArfP   \@�Ar��   ArfP   Ar�P   \A�Ar��   Ar�P   ArHP   \@�Arm�   ArHP   Ar�P   \@�Ar��   Ar�P   Ar*P   \@�ArO�   Ar*P   Ar�P   \A�Ar��   Ar�P   ArP   \@�Ar1�   ArP   Ar}P   \@�Ar��   Ar}P   Ar�P   \@�Ar�   Ar�P   Ar_P   \A�Ar��   Ar_P   Ar�P   \@�Ar��   Ar�P   ArAP   \@�Arf�   ArAP   Ar�P   \@�Ar��   Ar�P   Ar!#P   \BBAr!H�   Ar!#P   Ar#�P   \@�Ar#��   Ar#�P   Ar&P   \@�Ar&*�   Ar&P   Ar(vP   \@�Ar(��   Ar(vP   Ar*�P   \A�Ar+�   Ar*�P   Ar-XP   \@�Ar-}�   Ar-XP   Ar/�P   \@�Ar/��   Ar/�P   Ar2:P   \@�Ar2_�   Ar2:P   Ar4�P   \A�Ar4��   Ar4�P   Ar7P   \@�Ar7A�   Ar7P   Ar9�P   \@�Ar9��   Ar9�P   Ar;�P   \@�Ar<#�   Ar;�P   Ar>oP   \A�Ar>��   Ar>oP   Ar@�P   \@�ArA�   Ar@�P   ArCQP   \@�ArCv�   ArCQP   ArE�P   \@�ArE��   ArE�P   ArH3P   \A�ArHX�   ArH3P   ArJ�P   \@�ArJ��   ArJ�P   ArMP   \@�ArM:�   ArMP   ArO�P   \@�ArO��   ArO�P   ArQ�P   \A�ArR�   ArQ�P   ArThP   \@�ArT��   ArThP   ArV�P   \@�ArV��   ArV�P   ArYJP   \@�ArYo�   ArYJP   Ar[�P   \A�Ar[��   Ar[�P   Ar^,P   \@�Ar^Q�   Ar^,P   Ar`�P   \@�Ar`��   Ar`�P   ArcP   \@�Arc3�   ArcP   AreP   \A�Are��   AreP   Arg�P   \@�Arh�   Arg�P   ArjaP   \@�Arj��   ArjaP   Arl�P   \@�Arl��   Arl�P   AroCP   \A�Aroh�   AroCP   Arq�P   \@�Arq��   Arq�P   Art%P   \@�ArtJ�   Art%P   Arv�P   \@�Arv��   Arv�P   AryP   \A�Ary,�   AryP   Ar{xP   \@�Ar{��   Ar{xP   Ar}�P   \@�Ar~�   Ar}�P   Ar�ZP   \@�Ar��   Ar�ZP   Ar��P   \A�Ar���   Ar��P   Ar�<P   \@�Ar�a�   Ar�<P   Ar��P   \@�Ar���   Ar��P   Ar�P   \@�Ar�C�   Ar�P   Ar��P   \A�Ar���   Ar��P   Ar� P   \@�Ar�%�   Ar� P   Ar�qP   \@�Ar���   Ar�qP   Ar��P   \@�Ar��   Ar��P   Ar�SP   \A�Ar�x�   Ar�SP   Ar��P   \@�Ar���   Ar��P   Ar�5P   \@�Ar�Z�   Ar�5P   Ar��P   \@�Ar���   Ar��P   Ar�P   \A�Ar�<�   Ar�P   Ar��P   \@�Ar���   Ar��P   Ar��P   \@�Ar��   Ar��P   Ar�jP   \@�Ar���   Ar�jP   Ar��P   \A�Ar� �   Ar��P   Ar�LP   \@�Ar�q�   Ar�LP   Ar��P   \@�Ar���   Ar��P   Ar�.P   \@�Ar�S�   Ar�.P   Ar��P   \A�Ar���   Ar��P   Ar�P   \@�Ar�5�   Ar�P   Ar��P   \@�Ar���   Ar��P   Ar��P   \@�Ar��   Ar��P   Ar�cP   \A�Ar���   Ar�cP   Ar��P   \@�Ar���   Ar��P   Ar�EP   \@�Ar�j�   Ar�EP   ArĶP   \@�Ar���   ArĶP   Ar�'P   \A�Ar�L�   Ar�'P   ArɘP   \@�Arɽ�   ArɘP   Ar�	P   \@�Ar�.�   Ar�	P   Ar�zP   \@�ArΟ�   Ar�zP   Ar��P   \A�Ar��   Ar��P   Ar�\P   \@�ArӁ�   Ar�\P   Ar��P   \@�Ar���   Ar��P   Ar�>P   \@�Ar�c�   Ar�>P   ArگP   \A�Ar���   ArگP   Ar� P   \@�Ar�E�   Ar� P   ArߑP   \@�Ar߶�   ArߑP   Ar�P   \@�Ar�'�   Ar�P   Ar�sP   \A�Ar��   Ar�sP   Ar��P   \@�Ar�	�   Ar��P   Ar�UP   \@�Ar�z�   Ar�UP   Ar��P   \@�Ar���   Ar��P   Ar�7P   \A�Ar�\�   Ar�7P   Ar�P   \@�Ar���   Ar�P   Ar�P   \@�Ar�>�   Ar�P   Ar��P   \@�Ar���   Ar��P   Ar��P   \A�Ar� �   Ar��P   Ar�lP   \@�Ar���   Ar�lP   Ar��P   \@�Ar��   Ar��P   Ar�NP   \@�Ar�s�   Ar�NP   As�P   \A�As��   As�P   As0P   \@�AsU�   As0P   As�P   \@�As��   As�P   As	P   \@�As	7�   As	P   As�P   \A�As��   As�P   As�P   \@�As�   As�P   AseP   \@�As��   AseP   As�P   \@�As��   As�P   AsGP   \A�Asl�   AsGP   As�P   \@�As��   As�P   As)P   \@�AsN�   As)P   As�P   \@�As��   As�P   AsP   \A�As0�   AsP   As!|P   \@�As!��   As!|P   As#�P   \@�As$�   As#�P   As&^P   \@�As&��   As&^P   As(�P   \A�As(��   As(�P   As+@P   \@�As+e�   As+@P   As-�P   \@�As-��   As-�P   As0"P   \@�As0G�   As0"P   As2�P   \A�As2��   As2�P   As5P   \@�As5)�   As5P   As7uP   \@�As7��   As7uP   As9�P   \@�As:�   As9�P   As<WP   \A�As<|�   As<WP   As>�P   \@�As>��   As>�P   AsA9P   \@�AsA^�   AsA9P   AsC�P   \@�AsC��   AsC�P   AsFP   \A�AsF@�   AsFP   AsH�P   \@�AsH��   AsH�P   AsJ�P   \@�AsK"�   AsJ�P   AsMnP   \@�AsM��   AsMnP   AsO�P   \A�AsP�   AsO�P   AsRPP   \@�AsRu�   AsRPP   AsT�P   \@�AsT��   AsT�P   AsW2P   \@�AsWW�   AsW2P   AsY�P   \A�AsY��   AsY�P   As\P   \@�As\9�   As\P   As^�P   \@�As^��   As^�P   As`�P   \@�Asa�   As`�P   AscgP   \A�Asc��   AscgP   Ase�P   \@�Ase��   Ase�P   AshIP   \@�Ashn�   AshIP   Asj�P   \@�Asj��   Asj�P   Asm+P   \A�AsmP�   Asm+P   Aso�P   \@�Aso��   Aso�P   AsrP   \@�Asr2�   AsrP   Ast~P   \@�Ast��   Ast~P   Asv�P   \A�Asw�   Asv�P   Asy`P   \@�Asy��   Asy`P   As{�P   \@�As{��   As{�P   As~BP   \@�As~g�   As~BP   As��P   \A�As���   As��P   As�$P   \@�As�I�   As�$P   As��P   \@�As���   As��P   As�P   \@�As�+�   As�P   As�wP   \A�As���   As�wP   As��P   \@�As��   As��P   As�YP   \@�As�~�   As�YP   As��P   \@�As���   As��P   As�;P   \A�As�`�   As�;P   As��P   \@�As���   As��P   As�P   \@�As�B�   As�P   As��P   \@�As���   As��P   As��P   \A�As�$�   As��P   As�pP   \@�As���   As�pP   As��P   \@�As��   As��P   As�RP   \@�As�w�   As�RP   As��P   \A�As���   As��P   As�4P   \@�As�Y�   As�4P   As��P   \@�As���   As��P   As�P   \@�As�;�   As�P   As��P   \A�As���   As��P   As��P   \@�As��   As��P   As�iP   \@�As���   As�iP   As��P   \@�As���   As��P   As�KP   \A�As�p�   As�KP   As��P   \@�As���   As��P   As�-P   \@�As�R�   As�-P   AsP   \@�As���   AsP   As�P   \A�As�4�   As�P   AsǀP   \@�Asǥ�   AsǀP   As��P   \@�As��   As��P   As�bP   \@�Aṡ�   As�bP   As��P   \A�As���   As��P   As�DP   \@�As�i�   As�DP   AsӵP   \@�As���   AsӵP   As�&P   \@�As�K�   As�&P   AsؗP   \A�Asؼ�   AsؗP   As�P   \@�As�-�   As�P   As�yP   \@�Asݞ�   As�yP   As��P   \@�As��   As��P   As�[P   \A�As��   As�[P   As��P   \@�As���   As��P   As�=P   \@�As�b�   As�=P   As�P   \@�As���   As�P   As�P   \A�As�D�   As�P   As�P   \@�As��   As�P   As�P   \@�As�&�   As�P   As�rP   \@�As��   As�rP   As��P   \A�As��   As��P   As�TP   \@�As�y�   As�TP   As��P   \@�As���   As��P   As�6P   \@�As�[�   As�6P   As��P   \A�As���   As��P   AtP   \@�At=�   AtP   At�P   \@�At��   At�P   At�P   \@�At�   At�P   At	kP   \BBAt	��   At	kP   At�P   \@�At�   At�P   AtMP   \@�Atr�   AtMP   At�P   \@�At��   At�P   At/P   \A�AtT�   At/P   At�P   \@�At��   At�P   AtP   \@�At6�   AtP   At�P   \@�At��   At�P   At�P   \A�At�   At�P   AtdP   \@�At��   AtdP   At!�P   \@�At!��   At!�P   At$FP   \@�At$k�   At$FP   At&�P   \A�At&��   At&�P   At)(P   \@�At)M�   At)(P   At+�P   \@�At+��   At+�P   At.
P   \@�At./�   At.
P   At0{P   \A�At0��   At0{P   At2�P   \@�At3�   At2�P   At5]P   \@�At5��   At5]P   At7�P   \@�At7��   At7�P   At:?P   \A�At:d�   At:?P   At<�P   \@�At<��   At<�P   At?!P   \@�At?F�   At?!P   AtA�P   \@�AtA��   AtA�P   AtDP   \A�AtD(�   AtDP   AtFtP   \@�AtF��   AtFtP   AtH�P   \@�AtI
�   AtH�P   AtKVP   \@�AtK{�   AtKVP   AtM�P   \A�AtM��   AtM�P   AtP8P   \@�AtP]�   AtP8P   AtR�P   \@�AtR��   AtR�P   AtUP   \@�AtU?�   AtUP   AtW�P   \A�AtW��   AtW�P   AtY�P   \@�AtZ!�   AtY�P   At\mP   \@�At\��   At\mP   At^�P   \@�At_�   At^�P   AtaOP   \A�Atat�   AtaOP   Atc�P   \@�Atc��   Atc�P   Atf1P   \@�AtfV�   Atf1P   Ath�P   \@�Ath��   Ath�P   AtkP   \A�Atk8�   AtkP   Atm�P   \@�Atm��   Atm�P   Ato�P   \@�Atp�   Ato�P   AtrfP   \@�Atr��   AtrfP   Att�P   \A�Att��   Att�P   AtwHP   \@�Atwm�   AtwHP   Aty�P   \@�Aty��   Aty�P   At|*P   \@�At|O�   At|*P   At~�P   \A�At~��   At~�P   At�P   \@�At�1�   At�P   At�}P   \@�At���   At�}P   At��P   \@�At��   At��P   At�_P   \A�At���   At�_P   At��P   \@�At���   At��P   At�AP   \@�At�f�   At�AP   At��P   \@�At���   At��P   At�#P   \A�At�H�   At�#P   At��P   \@�At���   At��P   At�P   \@�At�*�   At�P   At�vP   \@�At���   At�vP   At��P   \A�At��   At��P   At�XP   \@�At�}�   At�XP   At��P   \@�At���   At��P   At�:P   \@�At�_�   At�:P   At��P   \A�At���   At��P   At�P   \@�At�A�   At�P   At��P   \@�At���   At��P   At��P   \@�At�#�   At��P   At�oP   \A�At���   At�oP   At��P   \@�At��   At��P   At�QP   \@�At�v�   At�QP   At��P   \@�At���   At��P   At�3P   \A�At�X�   At�3P   At��P   \@�At���   At��P   At�P   \@�At�:�   At�P   At��P   \@�At���   At��P   At��P   \A�At��   At��P   At�hP   \@�Atō�   At�hP   At��P   \@�At���   At��P   At�JP   \@�At�o�   At�JP   At̻P   \A�At���   At̻P   At�,P   \@�At�Q�   At�,P   AtѝP   \@�At���   AtѝP   At�P   \@�At�3�   At�P   At�P   \A�At֤�   At�P   At��P   \@�At��   At��P   At�aP   \@�Atۆ�   At�aP   At��P   \@�At���   At��P   At�CP   \A�At�h�   At�CP   At�P   \@�At���   At�P   At�%P   \@�At�J�   At�%P   At�P   \@�At��   At�P   At�P   \A�At�,�   At�P   At�xP   \@�At��   At�xP   At��P   \@�At��   At��P   At�ZP   \@�At��   At�ZP   At��P   \A�At���   At��P   At�<P   \@�At�a�   At�<P   At��P   \@�At���   At��P   At�P   \@�At�C�   At�P   At��P   \BBAt���   At��P   Au  P   \@�Au %�   Au  P   AuqP   \@�Au��   AuqP   Au�P   \@�Au�   Au�P   AuSP   \A�Aux�   AuSP   Au	�P   \@�Au	��   Au	�P   Au5P   \@�AuZ�   Au5P   Au�P   \@�Au��   Au�P   AuP   \A�Au<�   AuP   Au�P   \@�Au��   Au�P   Au�P   \@�Au�   Au�P   AujP   \@�Au��   AujP   Au�P   \A�Au �   Au�P   AuLP   \@�Auq�   AuLP   Au�P   \@�Au��   Au�P   Au".P   \@�Au"S�   Au".P   Au$�P   \A�Au$��   Au$�P   Au'P   \@�Au'5�   Au'P   Au)�P   \@�Au)��   Au)�P   Au+�P   \@�Au,�   Au+�P   Au.cP   \A�Au.��   Au.cP   Au0�P   \@�Au0��   Au0�P   Au3EP   \@�Au3j�   Au3EP   Au5�P   \@�Au5��   Au5�P   Au8'P   \A�Au8L�   Au8'P   Au:�P   \@�Au:��   Au:�P   Au=	P   \@�Au=.�   Au=	P   Au?zP   \@�Au?��   Au?zP   AuA�P   \A�AuB�   AuA�P   AuD\P   \@�AuD��   AuD\P   AuF�P   \@�AuF��   AuF�P   AuI>P   \@�AuIc�   AuI>P   AuK�P   \A�AuK��   AuK�P   AuN P   \@�AuNE�   AuN P   AuP�P   \@�AuP��   AuP�P   AuSP   \@�AuS'�   AuSP   AuUsP   \A�AuU��   AuUsP   AuW�P   \@�AuX	�   AuW�P   AuZUP   \@�AuZz�   AuZUP   Au\�P   \@�Au\��   Au\�P   Au_7P   \A�Au_\�   Au_7P   Aua�P   \@�Aua��   Aua�P   AudP   \@�Aud>�   AudP   Auf�P   \@�Auf��   Auf�P   Auh�P   \A�Aui �   Auh�P   AuklP   \@�Auk��   AuklP   Aum�P   \@�Aun�   Aum�P   AupNP   \@�Aups�   AupNP   Aur�P   \A�Aur��   Aur�P   Auu0P   \@�AuuU�   Auu0P   Auw�P   \@�Auw��   Auw�P   AuzP   \@�Auz7�   AuzP   Au|�P   \A�Au|��   Au|�P   Au~�P   \@�Au�   Au~�P   Au�eP   \@�Au���   Au�eP   Au��P   \@�Au���   Au��P   Au�GP   \A�Au�l�   Au�GP   Au��P   \@�Au���   Au��P   Au�)P   \@�Au�N�   Au�)P   Au��P   \@�Au���   Au��P   Au�P   \A�Au�0�   Au�P   Au�|P   \@�Au���   Au�|P   Au��P   \@�Au��   Au��P   Au�^P   \@�Au���   Au�^P   Au��P   \A�Au���   Au��P   Au�@P   \@�Au�e�   Au�@P   Au��P   \@�Au���   Au��P   Au�"P   \@�Au�G�   Au�"P   Au��P   \A�Au���   Au��P   Au�P   \@�Au�)�   Au�P   Au�uP   \@�Au���   Au�uP   Au��P   \@�Au��   Au��P   Au�WP   \A�Au�|�   Au�WP   Au��P   \@�Au���   Au��P   Au�9P   \@�Au�^�   Au�9P   Au��P   \@�Au���   Au��P   Au�P   \A�Au�@�   Au�P   Au��P   \@�Au���   Au��P   Au��P   \@�Au�"�   Au��P   Au�nP   \@�Au���   Au�nP   Au��P   \A�Au��   Au��P   Au�PP   \@�Au�u�   Au�PP   Au��P   \@�Au���   Au��P   Au�2P   \@�Au�W�   Au�2P   AuʣP   \A�Au���   AuʣP   Au�P   \@�Au�9�   Au�P   AuυP   \@�AuϪ�   AuυP   Au��P   \@�Au��   Au��P   Au�gP   \A�AuԌ�   Au�gP   Au��P   \@�Au���   Au��P   Au�IP   \@�Au�n�   Au�IP   AuۺP   \@�Au���   AuۺP   Au�+P   \A�Au�P�   Au�+P   Au��P   \@�Au���   Au��P   Au�P   \@�Au�2�   Au�P   Au�~P   \@�Au��   Au�~P   Au��P   \A�Au��   Au��P   Au�`P   \@�Au��   Au�`P   Au��P   \@�Au���   Au��P   Au�BP   \@�Au�g�   Au�BP   Au�P   \BBAu���   Au�P   Au�$P   \@�Au�I�   Au�$P   Au��P   \@�Au���   Au��P   Au�P   \@�Au�+�   Au�P   Au�wP   \A�Au���   Au�wP   Au��P   \@�Au��   Au��P   Av YP   \@�Av ~�   Av YP   Av�P   \@�Av��   Av�P   Av;P   \A�Av`�   Av;P   Av�P   \@�Av��   Av�P   Av
P   \@�Av
B�   Av
P   Av�P   \@�Av��   Av�P   Av�P   \A�Av$�   Av�P   AvpP   \@�Av��   AvpP   Av�P   \@�Av�   Av�P   AvRP   \@�Avw�   AvRP   Av�P   \A�Av��   Av�P   Av4P   \@�AvY�   Av4P   Av�P   \@�Av��   Av�P   Av P   \@�Av ;�   Av P   Av"�P   \A�Av"��   Av"�P   Av$�P   \@�Av%�   Av$�P   Av'iP   \@�Av'��   Av'iP   Av)�P   \@�Av)��   Av)�P   Av,KP   \A�Av,p�   Av,KP   Av.�P   \@�Av.��   Av.�P   Av1-P   \@�Av1R�   Av1-P   Av3�P   \@�Av3��   Av3�P   Av6P   \A�Av64�   Av6P   Av8�P   \@�Av8��   Av8�P   Av:�P   \@�Av;�   Av:�P   Av=bP   \@�Av=��   Av=bP   Av?�P   \A�Av?��   Av?�P   AvBDP   \@�AvBi�   AvBDP   AvD�P   \@�AvD��   AvD�P   AvG&P   \@�AvGK�   AvG&P   AvI�P   \A�AvI��   AvI�P   AvLP   \@�AvL-�   AvLP   AvNyP   \@�AvN��   AvNyP   AvP�P   \@�AvQ�   AvP�P   AvS[P   \A�AvS��   AvS[P   AvU�P   \@�AvU��   AvU�P   AvX=P   \@�AvXb�   AvX=P   AvZ�P   \@�AvZ��   AvZ�P   Av]P   \A�Av]D�   Av]P   Av_�P   \@�Av_��   Av_�P   AvbP   \@�Avb&�   AvbP   AvdrP   \@�Avd��   AvdrP   Avf�P   \A�Avg�   Avf�P   AviTP   \@�