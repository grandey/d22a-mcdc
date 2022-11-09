CDF   �   
      time       bnds      lon       lat          8   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-CM6-1 (2017):  aerosol: prescribed monthly fields computed by TACTIC_v2 scheme atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: OZL_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) seaIce: Gelato 6.1     institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    name      �/scratch/work/voldoire/outputs/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/18500101/rsut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_%start_date%-%end_date%      creation_date         2018-06-20T08:40:01Z   description       CMIP6 historical   title         <CNRM-CM6-1 model output prepared for CMIP6 / CMIP historical   activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.10   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-CM6-1.historical.none.r1i1p1f2      grid      ldata regridded to a T127 gaussian grid (128x256 latlon) from a native atmosphere T127l reduced gaussian grid   
grid_label        gr     nominal_resolution        250 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      
CNRM-CM6-1     parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         atmos      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         
CNRM-CM6-1     source_type       AOGCM      sub_experiment_id         none   sub_experiment        none   table_id      Amon   variable_id       rsut   variant_label         r1i1p1f2   EXPID         CNRM-CM6-1_historical_r1i1p1f2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          d6225e658d7de0912fca2a4293dbe2a7   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      tracking_id       1hdl:21.14100/6ea708a5-5828-42b9-bf40-4c830e8bead0      history      6Wed Nov 09 18:59:59 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/CNRM-CM6-1_r1i1p1f2/rsut_CNRM-CM6-1_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CNRM-CM6-1_r1i1p1f2/CMIP6.ScenarioMIP.CNRM-CERFACS.CNRM-CM6-1.ssp245.r1i1p1f2.Amon.rsut.gr.v20190219/rsut_Amon_CNRM-CM6-1_ssp245_r1i1p1f2_gr_201501-210012.yearmean.mul.areacella_piControl_v20180814.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/CNRM-CM6-1_r1i1p1f2/rsut_CNRM-CM6-1_r1i1p1f2_ssp245.mergetime.nc
Wed Nov 09 18:59:58 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Amon.rsut.gr.v20180917/rsut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.yearmean.mul.areacella_historical_v20180917.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/CNRM-CM6-1_r1i1p1f2/rsut_CNRM-CM6-1_r1i1p1f2_historical.mergetime.nc
Fri Nov 04 06:23:37 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Amon.rsut.gr.v20180917/rsut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.fx.areacella.gr.v20180917/areacella_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Amon.rsut.gr.v20180917/rsut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.yearmean.mul.areacella_historical_v20180917.fldsum.nc
Mon Jul 30 11:14:06 2018: ncatted -O -a tracking_id,global,m,c,hdl:21.14100/6ea708a5-5828-42b9-bf40-4c830e8bead0 /scratch/work/voldoire/outputs/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/assembled/rsut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc
none     NCO       "4.5.5"    CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               h   	time_bnds                                 p   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               X   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               `   rsut                      standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   online_operation      average    interval_operation        900 s      interval_write        1 month    description       at the top of the atmosphere   positive      up     history       none   cell_measures         area: areacella             �                Aq���   Aq��P   Aq�P   [6��Aq�6�   Aq�P   Aq��P   [6XOAq���   Aq��P   Aq��P   [6�Aq��   Aq��P   Aq�dP   [6��Aq���   Aq�dP   Aq��P   [6�Aq���   Aq��P   Aq�FP   [6cUAq�k�   Aq�FP   Aq��P   [6��Aq���   Aq��P   Aq�(P   [6֎Aq�M�   Aq�(P   Aq��P   [6�cAq���   Aq��P   Aq�
P   [6��Aq�/�   Aq�
P   Aq�{P   [6�CAq���   Aq�{P   Aq��P   [6��Aq��   Aq��P   Aq�]P   [7C0AqĂ�   Aq�]P   Aq��P   [6�CAq���   Aq��P   Aq�?P   [73Aq�d�   Aq�?P   Aq˰P   [6��Aq���   Aq˰P   Aq�!P   [6̘Aq�F�   Aq�!P   AqВP   [6�!Aqз�   AqВP   Aq�P   [6�=Aq�(�   Aq�P   Aq�tP   [6a�Aqՙ�   Aq�tP   Aq��P   [6²Aq�
�   Aq��P   Aq�VP   [7HvAq�{�   Aq�VP   Aq��P   [6�:Aq���   Aq��P   Aq�8P   [6�Aq�]�   Aq�8P   Aq�P   [6��Aq���   Aq�P   Aq�P   [6�(Aq�?�   Aq�P   Aq�P   [6ڝAq��   Aq�P   Aq��P   [6�eAq�!�   Aq��P   Aq�mP   [6��Aq��   Aq�mP   Aq��P   [6��Aq��   Aq��P   Aq�OP   [7�Aq�t�   Aq�OP   Aq��P   [6ZoAq���   Aq��P   Aq�1P   [6�\Aq�V�   Aq�1P   Aq��P   [8c�Aq���   Aq��P   Aq�P   [9��Aq�8�   Aq�P   Aq��P   [7��Aq���   Aq��P   Aq��P   [7Aq��   Aq��P   ArfP   [7G�Ar��   ArfP   Ar�P   [7Ar��   Ar�P   ArHP   [6߃Arm�   ArHP   Ar�P   [7�Ar��   Ar�P   Ar*P   [6��ArO�   Ar*P   Ar�P   [6��Ar��   Ar�P   ArP   [7{yAr1�   ArP   Ar}P   [7>IAr��   Ar}P   Ar�P   [6��Ar�   Ar�P   Ar_P   [6��Ar��   Ar_P   Ar�P   [7/�Ar��   Ar�P   ArAP   [7'Arf�   ArAP   Ar�P   [6��Ar��   Ar�P   Ar!#P   [7C�Ar!H�   Ar!#P   Ar#�P   [7j&Ar#��   Ar#�P   Ar&P   [7��Ar&*�   Ar&P   Ar(vP   [9�Ar(��   Ar(vP   Ar*�P   [8�Ar+�   Ar*�P   Ar-XP   [7�UAr-}�   Ar-XP   Ar/�P   [7�VAr/��   Ar/�P   Ar2:P   [8zAr2_�   Ar2:P   Ar4�P   [7�pAr4��   Ar4�P   Ar7P   [7^�Ar7A�   Ar7P   Ar9�P   [7�wAr9��   Ar9�P   Ar;�P   [8�Ar<#�   Ar;�P   Ar>oP   [8 uAr>��   Ar>oP   Ar@�P   [8f�ArA�   Ar@�P   ArCQP   [7y	ArCv�   ArCQP   ArE�P   [7*wArE��   ArE�P   ArH3P   [7��ArHX�   ArH3P   ArJ�P   [89)ArJ��   ArJ�P   ArMP   [7�ArM:�   ArMP   ArO�P   [7ArO��   ArO�P   ArQ�P   [7��ArR�   ArQ�P   ArThP   [72NArT��   ArThP   ArV�P   [8:ArV��   ArV�P   ArYJP   [7z�ArYo�   ArYJP   Ar[�P   [7R�Ar[��   Ar[�P   Ar^,P   [7�tAr^Q�   Ar^,P   Ar`�P   [7�3Ar`��   Ar`�P   ArcP   [8�Arc3�   ArcP   AreP   [7N Are��   AreP   Arg�P   [7�,Arh�   Arg�P   ArjaP   [8)]Arj��   ArjaP   Arl�P   [7ڼArl��   Arl�P   AroCP   [7��Aroh�   AroCP   Arq�P   [7�Arq��   Arq�P   Art%P   [79�ArtJ�   Art%P   Arv�P   [8(
Arv��   Arv�P   AryP   [7�Ary,�   AryP   Ar{xP   [7XxAr{��   Ar{xP   Ar}�P   [7d�Ar~�   Ar}�P   Ar�ZP   [7Ar��   Ar�ZP   Ar��P   [7@Ar���   Ar��P   Ar�<P   [71�Ar�a�   Ar�<P   Ar��P   [7bXAr���   Ar��P   Ar�P   [7"�Ar�C�   Ar�P   Ar��P   [7v-Ar���   Ar��P   Ar� P   [7��Ar�%�   Ar� P   Ar�qP   [7IAr���   Ar�qP   Ar��P   [7W�Ar��   Ar��P   Ar�SP   [7��Ar�x�   Ar�SP   Ar��P   [7�3Ar���   Ar��P   Ar�5P   [7�zAr�Z�   Ar�5P   Ar��P   [7�WAr���   Ar��P   Ar�P   [7�Ar�<�   Ar�P   Ar��P   [7� Ar���   Ar��P   Ar��P   [7��Ar��   Ar��P   Ar�jP   [7�eAr���   Ar�jP   Ar��P   [7�fAr� �   Ar��P   Ar�LP   [7��Ar�q�   Ar�LP   Ar��P   [8O7Ar���   Ar��P   Ar�.P   [7��Ar�S�   Ar�.P   Ar��P   [7��Ar���   Ar��P   Ar�P   [8>HAr�5�   Ar�P   Ar��P   [7�uAr���   Ar��P   Ar��P   [9=�Ar��   Ar��P   Ar�cP   [8�eAr���   Ar�cP   Ar��P   [8��Ar���   Ar��P   Ar�EP   [8��Ar�j�   Ar�EP   ArĶP   [8/�Ar���   ArĶP   Ar�'P   [7��Ar�L�   Ar�'P   ArɘP   [8�Arɽ�   ArɘP   Ar�	P   [7��Ar�.�   Ar�	P   Ar�zP   [8(CArΟ�   Ar�zP   Ar��P   [83�Ar��   Ar��P   Ar�\P   [8EgArӁ�   Ar�\P   Ar��P   [7�Ar���   Ar��P   Ar�>P   [8X Ar�c�   Ar�>P   ArگP   [7��Ar���   ArگP   Ar� P   [8qfAr�E�   Ar� P   ArߑP   [8 �Ar߶�   ArߑP   Ar�P   [7�NAr�'�   Ar�P   Ar�sP   [8�Ar��   Ar�sP   Ar��P   [7��Ar�	�   Ar��P   Ar�UP   [8�mAr�z�   Ar�UP   Ar��P   [8gAr���   Ar��P   Ar�7P   [8c�Ar�\�   Ar�7P   Ar�P   [8%�Ar���   Ar�P   Ar�P   [85>Ar�>�   Ar�P   Ar��P   [7�eAr���   Ar��P   Ar��P   [7�)Ar� �   Ar��P   Ar�lP   [7`�Ar���   Ar�lP   Ar��P   [80�Ar��   Ar��P   Ar�NP   [8�*Ar�s�   Ar�NP   As�P   [:��As��   As�P   As0P   [9��AsU�   As0P   As�P   [7�hAs��   As�P   As	P   [7�yAs	7�   As	P   As�P   [7s9As��   As�P   As�P   [7�~As�   As�P   AseP   [7��As��   AseP   As�P   [7�;As��   As�P   AsGP   [7��Asl�   AsGP   As�P   [7�,As��   As�P   As)P   [7�VAsN�   As)P   As�P   [7L�As��   As�P   AsP   [8As0�   AsP   As!|P   [7�As!��   As!|P   As#�P   [7X3As$�   As#�P   As&^P   [7�bAs&��   As&^P   As(�P   [7"EAs(��   As(�P   As+@P   [7-�As+e�   As+@P   As-�P   [7n�As-��   As-�P   As0"P   [7�As0G�   As0"P   As2�P   [7As2��   As2�P   As5P   [6�&As5)�   As5P   As7uP   [6ޤAs7��   As7uP   As9�P   [7�As:�   As9�P   As<WP   [7>pAs<|�   As<WP   As>�P   [6�YAs>��   As>�P   AsA9P   [6�AsA^�   AsA9P   AsC�P   [6��AsC��   AsC�P   AsFP   [5��AsF@�   AsFP   AsH�P   [6�-AsH��   AsH�P   AsJ�P   [6��AsK"�   AsJ�P   AsMnP   [6CuAsM��   AsMnP   AsO�P   [63�AsP�   AsO�P   AsRPP   [5�AsRu�   AsRPP   AsT�P   [6�AsT��   AsT�P   AsW2P   [6R�AsWW�   AsW2P   AsY�P   [6=NAsY��   AsY�P   As\P   [6b}As\9�   As\P   As^�P   [5��As^��   As^�P   As`�P   [5�CAsa�   As`�P   AscgP   [55"Asc��   AscgP   Ase�P   [5��Ase��   Ase�P   AshIP   [5W�Ashn�   AshIP   Asj�P   [5��Asj��   Asj�P   Asm+P   [5ٱAsmP�   Asm+P   Aso�P   [5��Aso��   Aso�P   AsrP   [5�"Asr2�   AsrP   Ast~P   [5ͼAst��   Ast~P   Asv�P   [5x�Asw�   Asv�P   Asy`P   [5�_Asy��   Asy`P   As{�P   [5�/As{��   As{�P   As~BP   [5�}As~g�   As~BP   As��P   [5(�As���   As��P   As�$P   [5RCAs�I�   As�$P   As��P   [4��As���   As��P   As�P   [5As�+�   As�P   As�wP   [4ɝAs���   As�wP   As��P   [4��As��   As��P   As�YP   [5#As�~�   As�YP   As��P   [5�As���   As��P   As�;P   [4�As�`�   As�;P   As��P   [4;�As���   As��P   As�P   [4L�As�B�   As�P   As��P   [4F&As���   As��P   As��P   [4v�As�$�   As��P   As�pP   [42LAs���   As�pP   As��P   [4p As��   As��P   As�RP   [4c�As�w�   As�RP   As��P   [4$1As���   As��P   As�4P   [4]�As�Y�   As�4P   As��P   [3��As���   As��P   As�P   [4�As�;�   As�P   As��P   [3:As���   As��P   As��P   [3DpAs��   As��P   As�iP   [4!�As���   As�iP   As��P   [3�fAs���   As��P   As�KP   [3�As�p�   As�KP   As��P   [3�ZAs���   As��P   As�-P   [3�*As�R�   As�-P   AsP   [3�IAs���   AsP   As�P   [3�uAs�4�   As�P   AsǀP   [3�=Asǥ�   AsǀP   As��P   [3��As��   As��P   As�bP   [3j�Aṡ�   As�bP   As��P   [2�As���   As��P   As�DP   [3\As�i�   As�DP   AsӵP   [3h�As���   AsӵP   As�&P   [3t�As�K�   As�&P   AsؗP   [3m�Asؼ�   AsؗP   As�P   [3�As�-�   As�P   As�yP   [3r�Asݞ�   As�yP   As��P   [2��As��   As��P   As�[P   [3As��   As�[P   As��P   [2��As���   As��P   As�=P   [2��As�b�   As�=P   As�P   [3T~As���   As�P   As�P   [2�As�D�   As�P   As�P   [2�mAs��   As�P   As�P   [2�JAs�&�   As�P   As�rP   [2D�As��   As�rP   As��P   [2i�As��   As��P   As�TP   [2R�As�y�   As�TP   As��P   [2,6As���   As��P   As�6P   [1�As�[�   As�6P   As��P   [2��As���   As��P   AtP   [2�4At=�   AtP   At�P   [2S�At��   At�P   At�P   [2��At�   At�P   At	kP   [2hQ