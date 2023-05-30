CDF   �   
      time       bnds      lon       lat          0   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �UKESM1.0-LL (2018): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: UKCA-StratTrop
land: JULES-ES-1.0
landIce: none
ocean: NEMO-HadGEM3-GO6.0 (eORCA1 tripolar primarily 1 deg with meridional refinement down to 1/3 degree in the tropics; 360 x 330 longitude/latitude; 75 levels; top grid cell 0-1 m)
ocnBgchem: MEDUSA2
seaIce: CICE-HadGEM3-GSI8 (eORCA1 tripolar primarily 1 deg; 360 x 330 longitude/latitude)   institution       BMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK     activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A�        creation_date         2019-06-24T12:29:25Z   
cv_version        6.2.20.1   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Nhttps://furtherinfo.es-doc.org/CMIP6.MOHC.UKESM1-0-LL.historical.none.r1i1p1f2     grid      -Native N96 grid; 192 x 144 longitude/latitude      
grid_label        gn     history      	�Tue May 30 16:58:46 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/UKESM1-0-LL_r1i1p1f2/rsut_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp126.r1i1p1f2.Amon.rsut.gn.v20190708/rsut_Amon_UKESM1-0-LL_ssp126_r1i1p1f2_gn_201501-204912.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp126.r1i1p1f2.Amon.rsut.gn.v20190708/rsut_Amon_UKESM1-0-LL_ssp126_r1i1p1f2_gn_205001-210012.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/UKESM1-0-LL_r1i1p1f2/rsut_UKESM1-0-LL_r1i1p1f2_ssp126.mergetime.nc
Tue May 30 16:58:46 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/UKESM1-0-LL_r1i1p1f2/rsut_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc
Fri Nov 04 07:36:13 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.piControl.r1i1p1f2.fx.areacella.gn.v20190705/areacella_fx_UKESM1-0-LL_piControl_r1i1p1f2_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.yearmean.mul.areacella_piControl_v20190705.fldsum.nc
2019-06-24T12:18:04Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
2019-06-24T12:17:53Z MIP Convert v1.1.0, Python v2.7.12, Iris v1.13.0, Numpy v1.13.3, netcdftime v1.4.1.   initialization_index            institution_id        MOHC   mip_era       CMIP6      mo_runid      u-bc179    nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      UKESM1-0-LL    parent_time_units         days since 1850-01-01-00-00-00     parent_variant_label      r1i1p1f2   physics_index               product       model-output   realization_index               realm         atmos      	source_id         UKESM1-0-LL    source_type       AOGCM AER BGC CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(13 December 2018) MD5:2b12b5db6db112aa8b8b0d6c1645b121      title         %UKESM1-0-LL output prepared for CMIP6      variable_id       rsut   variant_label         r1i1p1f2   license      XCMIP6 model data produced by the Met Office Hadley Centre is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https://ukesm.ac.uk/cmip6. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   cmor_version      3.4.0      tracking_id       1hdl:21.14100/262e3690-fad7-4892-9b1b-a26f6034ab1d      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   
   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment      umo: For instantaneous outputs, this diagnostic represents an average over the radiation time step using the state of the atmosphere (T,q,clouds) from the first dynamics step within that interval. The time coordinate is the start of the radiation time step interval, so the value for t(N) is the average from t(N) to t(N+1)., CMIP_table_comment: at the top of the atmosphere      original_name         $mo: (stash: m01s01i208, lbproc: 128)   cell_measures         area: areacella    history       u2019-06-24T12:29:25Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Aq���   Aq��P   Aq�P   [3��Aq�6�   Aq�P   Aq��P   [4�Aq���   Aq��P   Aq��P   [4��Aq��   Aq��P   Aq�dP   [3EAq���   Aq�dP   Aq��P   [4bAq���   Aq��P   Aq�FP   [4$SAq�k�   Aq�FP   Aq��P   [3��Aq���   Aq��P   Aq�(P   [3��Aq�M�   Aq�(P   Aq��P   [3+�Aq���   Aq��P   Aq�
P   [3�nAq�/�   Aq�
P   Aq�{P   [3��Aq���   Aq�{P   Aq��P   [5;LAq��   Aq��P   Aq�]P   [5SAqĂ�   Aq�]P   Aq��P   [3��Aq���   Aq��P   Aq�?P   [4�BAq�d�   Aq�?P   Aq˰P   [4k+Aq���   Aq˰P   Aq�!P   [3��Aq�F�   Aq�!P   AqВP   [4��Aqз�   AqВP   Aq�P   [4�9Aq�(�   Aq�P   Aq�tP   [52�Aqՙ�   Aq�tP   Aq��P   [5 Aq�
�   Aq��P   Aq�VP   [4Aq�{�   Aq�VP   Aq��P   [3�Aq���   Aq��P   Aq�8P   [4E_Aq�]�   Aq�8P   Aq�P   [4L�Aq���   Aq�P   Aq�P   [4v,Aq�?�   Aq�P   Aq�P   [3��Aq��   Aq�P   Aq��P   [3��Aq�!�   Aq��P   Aq�mP   [3ڛAq��   Aq�mP   Aq��P   [4�,Aq��   Aq��P   Aq�OP   [41Aq�t�   Aq�OP   Aq��P   [4֖Aq���   Aq��P   Aq�1P   [3��Aq�V�   Aq�1P   Aq��P   [6�UAq���   Aq��P   Aq�P   [:vNAq�8�   Aq�P   Aq��P   [6��Aq���   Aq��P   Aq��P   [5�Aq��   Aq��P   ArfP   [4��Ar��   ArfP   Ar�P   [4h�Ar��   Ar�P   ArHP   [3�vArm�   ArHP   Ar�P   [4VAr��   Ar�P   Ar*P   [4w;ArO�   Ar*P   Ar�P   [4�/Ar��   Ar�P   ArP   [4�-Ar1�   ArP   Ar}P   [4L�Ar��   Ar}P   Ar�P   [3�jAr�   Ar�P   Ar_P   [4WAr��   Ar_P   Ar�P   [4ygAr��   Ar�P   ArAP   [4sArf�   ArAP   Ar�P   [4�Ar��   Ar�P   Ar!#P   [4�Ar!H�   Ar!#P   Ar#�P   [4�Ar#��   Ar#�P   Ar&P   [4�qAr&*�   Ar&P   Ar(vP   [7*Ar(��   Ar(vP   Ar*�P   [5�]Ar+�   Ar*�P   Ar-XP   [5�`Ar-}�   Ar-XP   Ar/�P   [4��Ar/��   Ar/�P   Ar2:P   [4�Ar2_�   Ar2:P   Ar4�P   [4�Ar4��   Ar4�P   Ar7P   [4�NAr7A�   Ar7P   Ar9�P   [5ISAr9��   Ar9�P   Ar;�P   [4��Ar<#�   Ar;�P   Ar>oP   [5�Ar>��   Ar>oP   Ar@�P   [5�QArA�   Ar@�P   ArCQP   [4�?ArCv�   ArCQP   ArE�P   [4}ArE��   ArE�P   ArH3P   [4�ArHX�   ArH3P   ArJ�P   [5&|ArJ��   ArJ�P   ArMP   [4�
ArM:�   ArMP   ArO�P   [4��ArO��   ArO�P   ArQ�P   [5f�ArR�   ArQ�P   ArThP   [5-3ArT��   ArThP   ArV�P   [4�ArV��   ArV�P   ArYJP   [5lHArYo�   ArYJP   Ar[�P   [4�Ar[��   Ar[�P   Ar^,P   [4ߘAr^Q�   Ar^,P   Ar`�P   [4�oAr`��   Ar`�P   ArcP   [5R�Arc3�   ArcP   AreP   [5��Are��   AreP   Arg�P   [5R�Arh�   Arg�P   ArjaP   [5? Arj��   ArjaP   Arl�P   [5HtArl��   Arl�P   AroCP   [5=Aroh�   AroCP   Arq�P   [5n�Arq��   Arq�P   Art%P   [57ArtJ�   Art%P   Arv�P   [4��Arv��   Arv�P   AryP   [5+Ary,�   AryP   Ar{xP   [5ɢAr{��   Ar{xP   Ar}�P   [5V2Ar~�   Ar}�P   Ar�ZP   [5�-Ar��   Ar�ZP   Ar��P   [5�GAr���   Ar��P   Ar�<P   [5��Ar�a�   Ar�<P   Ar��P   [5��Ar���   Ar��P   Ar�P   [5�vAr�C�   Ar�P   Ar��P   [5GLAr���   Ar��P   Ar� P   [5��Ar�%�   Ar� P   Ar�qP   [5Q Ar���   Ar�qP   Ar��P   [5Q�Ar��   Ar��P   Ar�SP   [5~�Ar�x�   Ar�SP   Ar��P   [6�6Ar���   Ar��P   Ar�5P   [6BAr�Z�   Ar�5P   Ar��P   [5цAr���   Ar��P   Ar�P   [6cXAr�<�   Ar�P   Ar��P   [5�aAr���   Ar��P   Ar��P   [5Ar��   Ar��P   Ar�jP   [5��Ar���   Ar�jP   Ar��P   [5Z�Ar� �   Ar��P   Ar�LP   [5�Ar�q�   Ar�LP   Ar��P   [5�FAr���   Ar��P   Ar�.P   [5�Ar�S�   Ar�.P   Ar��P   [5��Ar���   Ar��P   Ar�P   [6R%Ar�5�   Ar�P   Ar��P   [5��Ar���   Ar��P   Ar��P   [:�9Ar��   Ar��P   Ar�cP   [99;Ar���   Ar�cP   Ar��P   [7�:Ar���   Ar��P   Ar�EP   [7%�Ar�j�   Ar�EP   ArĶP   [7��Ar���   ArĶP   Ar�'P   [853Ar�L�   Ar�'P   ArɘP   [7|�Arɽ�   ArɘP   Ar�	P   [7��Ar�.�   Ar�	P   Ar�zP   [8}{ArΟ�   Ar�zP   Ar��P   [8]Ar��   Ar��P   Ar�\P   [7�&ArӁ�   Ar�\P   Ar��P   [8ԥAr���   Ar��P   Ar�>P   [7}�Ar�c�   Ar�>P   ArگP   [7k�Ar���   ArگP   Ar� P   [7H�Ar�E�   Ar� P   ArߑP   [7+Ar߶�   ArߑP   Ar�P   [7��Ar�'�   Ar�P   Ar�sP   [7��Ar��   Ar�sP   Ar��P   [7J�Ar�	�   Ar��P   Ar�UP   [85yAr�z�   Ar�UP   Ar��P   [9*�Ar���   Ar��P   Ar�7P   [8�ZAr�\�   Ar�7P   Ar�P   [7��Ar���   Ar�P   Ar�P   [7zAr�>�   Ar�P   Ar��P   [7ۭAr���   Ar��P   Ar��P   [6�(Ar� �   Ar��P   Ar�lP   [6�Ar���   Ar�lP   Ar��P   [6�[Ar��   Ar��P   Ar�NP   [9|SAr�s�   Ar�NP   As�P   [<ajAs��   As�P   As0P   [9*�AsU�   As0P   As�P   [77As��   As�P   As	P   [6��As	7�   As	P   As�P   [5�LAs��   As�P   As�P   [4��As�   As�P   AseP   [5�As��   AseP   As�P   [4�'As��   As�P   AsGP   [5.Asl�   AsGP   As�P   [4њAs��   As�P   As)P   [4��AsN�   As)P   As�P   [4�|As��   As�P   AsP   [4��As0�   AsP   As!|P   [4�(As!��   As!|P   As#�P   [5E�As$�   As#�P   As&^P   [4�5As&��   As&^P   As(�P   [3�@As(��   As(�P   As+@P   [3�EAs+e�   As+@P   As-�P   [3VlAs-��   As-�P   As0"P   [3b?As0G�   As0"P   As2�P   [3^�As2��   As2�P   As5P   [3K�As5)�   As5P   As7uP   [3��As7��   As7uP   As9�P   [3�As:�   As9�P   As<WP   [2�As<|�   As<WP   As>�P   [2T�As>��   As>�P   AsA9P   [1ŎAsA^�   AsA9P   AsC�P   [1�AsC��   AsC�P   AsFP   [1� AsF@�   AsFP   AsH�P   [1EAsH��   AsH�P   AsJ�P   [1'�AsK"�   AsJ�P   AsMnP   [1I�AsM��   AsMnP   AsO�P   [0�AsP�   AsO�P   AsRPP   [1hAsRu�   AsRPP   AsT�P   [0��AsT��   AsT�P   AsW2P   [0p
AsWW�   AsW2P   AsY�P   [0�`AsY��   AsY�P   As\P   [0��As\9�   As\P   As^�P   [0�MAs^��   As^�P   As`�P   [0iAsa�   As`�P   AscgP   [0�Asc��   AscgP   Ase�P   [/� Ase��   Ase�P   AshIP   [/�Ashn�   AshIP   Asj�P   [/�CAsj��   Asj�P   Asm+P   [/Q�AsmP�   Asm+P   Aso�P   [/P#Aso��   Aso�P   AsrP   [/Asr2�   AsrP   Ast~P   [.��Ast��   Ast~P   Asv�P   [/��Asw�   Asv�P   Asy`P   [.��Asy��   Asy`P   As{�P   [.y�As{��   As{�P   As~BP   [.otAs~g�   As~BP   As��P   [.B�As���   As��P   As�$P   [. �As�I�   As�$P   As��P   [.�As���   As��P   As�P   [.d�As�+�   As�P   As�wP   [.^�As���   As�wP   As��P   [.aAs��   As��P   As�YP   [-��As�~�   As�YP   As��P   [.5�As���   As��P   As�;P   [-�]As�`�   As�;P   As��P   [-�IAs���   As��P   As�P   [-�!As�B�   As�P   As��P   [-�5As���   As��P   As��P   [-/�As�$�   As��P   As�pP   [-^�As���   As�pP   As��P   [,��As��   As��P   As�RP   [,�+As�w�   As�RP   As��P   [-(8As���   As��P   As�4P   [->As�Y�   As�4P   As��P   [-Y�As���   As��P   As�P   [-:As�;�   As�P   As��P   [.iAs���   As��P   As��P   [.�As��   As��P   As�iP   [,�kAs���   As�iP   As��P   [,��As���   As��P   As�KP   [-$�As�p�   As�KP   As��P   [,�jAs���   As��P   As�-P   [,%�As�R�   As�-P   AsP   [,��As���   AsP   As�P   [-.�As�4�   As�P   AsǀP   [,�Asǥ�   AsǀP   As��P   [,OqAs��   As��P   As�bP   [,csAṡ�   As�bP   As��P   [,��As���   As��P   As�DP   [,]�As�i�   As�DP   AsӵP   [,��As���   AsӵP   As�&P   [,iAs�K�   As�&P   AsؗP   [+�Asؼ�   AsؗP   As�P   [+�As�-�   As�P   As�yP   [+��Asݞ�   As�yP   As��P   [,8�As��   As��P   As�[P   [,��As��   As�[P   As��P   [,p�As���   As��P   As�=P   [,ŮAs�b�   As�=P   As�P   [,�nAs���   As�P   As�P   [,�As�D�   As�P   As�P   [,w�As��   As�P   As�P   [,3�As�&�   As�P   As�rP   [,��As��   As�rP   As��P   [+��As��   As��P   As�TP   [,6�As�y�   As�TP   As��P   [,eAs���   As��P   As�6P   [,zGAs�[�   As�6P   As��P   [,�(As���   As��P   AtP   [,�At=�   AtP   At�P   [,�$At��   At�P   At�P   [,y�At�   At�P   At	kP   [+��