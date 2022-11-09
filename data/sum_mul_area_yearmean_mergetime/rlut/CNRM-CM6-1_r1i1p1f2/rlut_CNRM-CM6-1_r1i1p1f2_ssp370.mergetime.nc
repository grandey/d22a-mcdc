CDF   �   
      time       bnds      lon       lat          8   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-CM6-1 (2017):  aerosol: prescribed monthly fields computed by TACTIC_v2 scheme atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: OZL_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) seaIce: Gelato 6.1     institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    name      �/scratch/work/voldoire/outputs/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/18500101/rlut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_%start_date%-%end_date%      creation_date         2018-06-20T08:40:03Z   description       CMIP6 historical   title         <CNRM-CM6-1 model output prepared for CMIP6 / CMIP historical   activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.10   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-CM6-1.historical.none.r1i1p1f2      grid      ldata regridded to a T127 gaussian grid (128x256 latlon) from a native atmosphere T127l reduced gaussian grid   
grid_label        gr     nominal_resolution        250 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      
CNRM-CM6-1     parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         atmos      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         
CNRM-CM6-1     source_type       AOGCM      sub_experiment_id         none   sub_experiment        none   table_id      Amon   variable_id       rlut   variant_label         r1i1p1f2   EXPID         CNRM-CM6-1_historical_r1i1p1f2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          d6225e658d7de0912fca2a4293dbe2a7   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      tracking_id       1hdl:21.14100/936fecb1-82d6-489c-8b6b-6cbcb2747e8d      history      3Wed Nov 09 19:00:47 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/CNRM-CM6-1_r1i1p1f2/rlut_CNRM-CM6-1_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CNRM-CM6-1_r1i1p1f2/CMIP6.ScenarioMIP.CNRM-CERFACS.CNRM-CM6-1.ssp370.r1i1p1f2.Amon.rlut.gr.v20190219/rlut_Amon_CNRM-CM6-1_ssp370_r1i1p1f2_gr_201501-210012.yearmean.mul.areacella_ssp370_v20190219.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/CNRM-CM6-1_r1i1p1f2/rlut_CNRM-CM6-1_r1i1p1f2_ssp370.mergetime.nc
Wed Nov 09 19:00:45 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Amon.rlut.gr.v20180917/rlut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.yearmean.mul.areacella_historical_v20180917.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/CNRM-CM6-1_r1i1p1f2/rlut_CNRM-CM6-1_r1i1p1f2_historical.mergetime.nc
Fri Nov 04 02:59:45 2022: cdo -O -s -fldsum -setattribute,rlut@units=W m-2 m2 -mul -yearmean -selname,rlut /Users/benjamin/Data/p22b/CMIP6/rlut/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Amon.rlut.gr.v20180917/rlut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.fx.areacella.gr.v20180917/areacella_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Amon.rlut.gr.v20180917/rlut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.yearmean.mul.areacella_historical_v20180917.fldsum.nc
Mon Jul 30 10:15:20 2018: ncatted -O -a tracking_id,global,m,c,hdl:21.14100/936fecb1-82d6-489c-8b6b-6cbcb2747e8d /scratch/work/voldoire/outputs/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/assembled/rlut_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc
none    NCO       "4.5.5"    CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rlut                      standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   online_operation      average    interval_operation        900 s      interval_write        1 month    description       Iat the top of the atmosphere (to be compared with satellite measurements)      positive      up     history       none   cell_measures         area: areacella             �                Aq���   Aq��P   Aq�P   [��Aq�6�   Aq�P   Aq��P   [���Aq���   Aq��P   Aq��P   [׹�Aq��   Aq��P   Aq�dP   [׿>Aq���   Aq�dP   Aq��P   [���Aq���   Aq��P   Aq�FP   [׿�Aq�k�   Aq�FP   Aq��P   [ף4Aq���   Aq��P   Aq�(P   [��Aq�M�   Aq�(P   Aq��P   [��{Aq���   Aq��P   Aq�
P   [��?Aq�/�   Aq�
P   Aq�{P   [���Aq���   Aq�{P   Aq��P   [ם�Aq��   Aq��P   Aq�]P   [�pAAqĂ�   Aq�]P   Aq��P   [י�Aq���   Aq��P   Aq�?P   [�WAq�d�   Aq�?P   Aq˰P   [׭�Aq���   Aq˰P   Aq�!P   [��Aq�F�   Aq�!P   AqВP   [��rAqз�   AqВP   Aq�P   [׺�Aq�(�   Aq�P   Aq�tP   [ץPAqՙ�   Aq�tP   Aq��P   [��Aq�
�   Aq��P   Aq�VP   [��Aq�{�   Aq�VP   Aq��P   [��4Aq���   Aq��P   Aq�8P   [׍�Aq�]�   Aq�8P   Aq�P   [��-Aq���   Aq�P   Aq�P   [�{�Aq�?�   Aq�P   Aq�P   [�o�Aq��   Aq�P   Aq��P   [�E�Aq�!�   Aq��P   Aq�mP   [ם,Aq��   Aq�mP   Aq��P   [�� Aq��   Aq��P   Aq�OP   [�!Aq�t�   Aq�OP   Aq��P   [׋�Aq���   Aq��P   Aq�1P   [׽�Aq�V�   Aq�1P   Aq��P   [׹�Aq���   Aq��P   Aq�P   [ַ Aq�8�   Aq�P   Aq��P   [�JAq���   Aq��P   Aq��P   [שAq��   Aq��P   ArfP   [�QiAr��   ArfP   Ar�P   [ײ�Ar��   Ar�P   ArHP   [�S�Arm�   ArHP   Ar�P   [�TAr��   Ar�P   Ar*P   [�}PArO�   Ar*P   Ar�P   [�t[Ar��   Ar�P   ArP   [�{LAr1�   ArP   Ar}P   [��Ar��   Ar}P   Ar�P   [׶�Ar�   Ar�P   Ar_P   [�zAr��   Ar_P   Ar�P   [ם�Ar��   Ar�P   ArAP   [רtArf�   ArAP   Ar�P   [�`�Ar��   Ar�P   Ar!#P   [�b�Ar!H�   Ar!#P   Ar#�P   [ז7Ar#��   Ar#�P   Ar&P   [�6ZAr&*�   Ar&P   Ar(vP   [�Ar(��   Ar(vP   Ar*�P   [�+�Ar+�   Ar*�P   Ar-XP   [��Ar-}�   Ar-XP   Ar/�P   [��Ar/��   Ar/�P   Ar2:P   [�?�Ar2_�   Ar2:P   Ar4�P   [�N,Ar4��   Ar4�P   Ar7P   [�%sAr7A�   Ar7P   Ar9�P   [�8LAr9��   Ar9�P   Ar;�P   [�GAr<#�   Ar;�P   Ar>oP   [��Ar>��   Ar>oP   Ar@�P   [�5SArA�   Ar@�P   ArCQP   [�f�ArCv�   ArCQP   ArE�P   [�:HArE��   ArE�P   ArH3P   [���ArHX�   ArH3P   ArJ�P   [�X�ArJ��   ArJ�P   ArMP   [�
�ArM:�   ArMP   ArO�P   [�TArO��   ArO�P   ArQ�P   [�dpArR�   ArQ�P   ArThP   [׊�ArT��   ArThP   ArV�P   [�!4ArV��   ArV�P   ArYJP   [�0�ArYo�   ArYJP   Ar[�P   [�PTAr[��   Ar[�P   Ar^,P   [�KwAr^Q�   Ar^,P   Ar`�P   [�"oAr`��   Ar`�P   ArcP   [�e�Arc3�   ArcP   AreP   [��Are��   AreP   Arg�P   [���Arh�   Arg�P   ArjaP   [�PArj��   ArjaP   Arl�P   [�I�Arl��   Arl�P   AroCP   [�6JAroh�   AroCP   Arq�P   [�Arq��   Arq�P   Art%P   [�[>ArtJ�   Art%P   Arv�P   [�
3Arv��   Arv�P   AryP   [��Ary,�   AryP   Ar{xP   [��bAr{��   Ar{xP   Ar}�P   [�2�Ar~�   Ar}�P   Ar�ZP   [�ljAr��   Ar�ZP   Ar��P   [�`dAr���   Ar��P   Ar�<P   [�UFAr�a�   Ar�<P   Ar��P   [�z�Ar���   Ar��P   Ar�P   [׀PAr�C�   Ar�P   Ar��P   [�4�Ar���   Ar��P   Ar� P   [�c+Ar�%�   Ar� P   Ar�qP   [א�Ar���   Ar�qP   Ar��P   [�'�Ar��   Ar��P   Ar�SP   [�.Ar�x�   Ar�SP   Ar��P   [׌�Ar���   Ar��P   Ar�5P   [�h8Ar�Z�   Ar�5P   Ar��P   [�HAr���   Ar��P   Ar�P   [�^(Ar�<�   Ar�P   Ar��P   [�G�Ar���   Ar��P   Ar��P   [��Ar��   Ar��P   Ar�jP   [�#Ar���   Ar�jP   Ar��P   [���Ar� �   Ar��P   Ar�LP   [�#�Ar�q�   Ar�LP   Ar��P   [ׄlAr���   Ar��P   Ar�.P   [�<`Ar�S�   Ar�.P   Ar��P   [�)_Ar���   Ar��P   Ar�P   [�GpAr�5�   Ar�P   Ar��P   [�0�Ar���   Ar��P   Ar��P   [��hAr��   Ar��P   Ar�cP   [�&Ar���   Ar�cP   Ar��P   [�tAr���   Ar��P   Ar�EP   [��Ar�j�   Ar�EP   ArĶP   [���Ar���   ArĶP   Ar�'P   [�(�Ar�L�   Ar�'P   ArɘP   [�o�Arɽ�   ArɘP   Ar�	P   [� Ar�.�   Ar�	P   Ar�zP   [�<�ArΟ�   Ar�zP   Ar��P   [�&�Ar��   Ar��P   Ar�\P   [�#ArӁ�   Ar�\P   Ar��P   [��Ar���   Ar��P   Ar�>P   [�=#Ar�c�   Ar�>P   ArگP   [���Ar���   ArگP   Ar� P   [��Ar�E�   Ar� P   ArߑP   [� Ar߶�   ArߑP   Ar�P   [��Ar�'�   Ar�P   Ar�sP   [�Z�Ar��   Ar�sP   Ar��P   [�X�Ar�	�   Ar��P   Ar�UP   [�%�Ar�z�   Ar�UP   Ar��P   [��3Ar���   Ar��P   Ar�7P   [ַ0Ar�\�   Ar�7P   Ar�P   [��Ar���   Ar�P   Ar�P   [��@Ar�>�   Ar�P   Ar��P   [�G Ar���   Ar��P   Ar��P   [�o�Ar� �   Ar��P   Ar�lP   [��Ar���   Ar�lP   Ar��P   [��Ar��   Ar��P   Ar�NP   [��Ar�s�   Ar�NP   As�P   [�5�As��   As�P   As0P   [��AAsU�   As0P   As�P   [���As��   As�P   As	P   [�MAs	7�   As	P   As�P   [��wAs��   As�P   As�P   [��As�   As�P   AseP   [��As��   AseP   As�P   [�As��   As�P   AsGP   [��Asl�   AsGP   As�P   [רCAs��   As�P   As)P   [��WAsN�   As)P   As�P   [��As��   As�P   AsP   [�As0�   AsP   As!|P   [�ƔAs!��   As!|P   As#�P   [��0As$�   As#�P   As&^P   [�;�As&��   As&^P   As(�P   [�"�As(��   As(�P   As+@P   [��yAs+e�   As+@P   As-�P   [�/]As-��   As-�P   As0"P   [��/As0G�   As0"P   As2�P   [�*[As2��   As2�P   As5P   [�?�As5)�   As5P   As7uP   [��As7��   As7uP   As9�P   [��As:�   As9�P   As<WP   [�
~As<|�   As<WP   As>�P   [���As>��   As>�P   AsA9P   [��BAsA^�   AsA9P   AsC�P   [��AsC��   AsC�P   AsFP   [���AsF@�   AsFP   AsH�P   [��AsH��   AsH�P   AsJ�P   [ּ�AsK"�   AsJ�P   AsMnP   [�:%AsM��   AsMnP   AsO�P   [�+�AsP�   AsO�P   AsRPP   [�`AsRu�   AsRPP   AsT�P   [�(AsT��   AsT�P   AsW2P   [��pAsWW�   AsW2P   AsY�P   [�'AsY��   AsY�P   As\P   [�ԏAs\9�   As\P   As^�P   [���As^��   As^�P   As`�P   [���Asa�   As`�P   AscgP   [�+�Asc��   AscgP   Ase�P   [���Ase��   Ase�P   AshIP   [��Ashn�   AshIP   Asj�P   [��8Asj��   Asj�P   Asm+P   [�6�AsmP�   Asm+P   Aso�P   [��;Aso��   Aso�P   AsrP   [���Asr2�   AsrP   Ast~P   [���Ast��   Ast~P   Asv�P   [�!ZAsw�   Asv�P   Asy`P   [�!Asy��   Asy`P   As{�P   [�`As{��   As{�P   As~BP   [��As~g�   As~BP   As��P   [�tzAs���   As��P   As�$P   [�g�As�I�   As�$P   As��P   [�QAs���   As��P   As�P   [�1.As�+�   As�P   As�wP   [�D�As���   As�wP   As��P   [�J}As��   As��P   As�YP   [��As�~�   As�YP   As��P   [�f�As���   As��P   As�;P   [�A�As�`�   As�;P   As��P   [��(As���   As��P   As�P   [�p�As�B�   As�P   As��P   [�W�As���   As��P   As��P   [�k�As�$�   As��P   As�pP   [�zAs���   As�pP   As��P   [ץ>As��   As��P   As�RP   [�p�As�w�   As�RP   As��P   [�"�As���   As��P   As�4P   [װ�As�Y�   As�4P   As��P   [�9�As���   As��P   As�P   [ׇ-As�;�   As�P   As��P   [ר�As���   As��P   As��P   [ױ�As��   As��P   As�iP   [�b�As���   As�iP   As��P   [׺�As���   As��P   As�KP   [׍ As�p�   As�KP   As��P   [�ÒAs���   As��P   As�-P   [׌�As�R�   As�-P   AsP   [���As���   AsP   As�P   [ױ�As�4�   As�P   AsǀP   [��Asǥ�   AsǀP   As��P   [ל�As��   As��P   As�bP   [״bAṡ�   As�bP   As��P   [���As���   As��P   As�DP   [��As�i�   As�DP   AsӵP   [��As���   AsӵP   As�&P   [��As�K�   As�&P   AsؗP   [��Asؼ�   AsؗP   As�P   [�HbAs�-�   As�P   As�yP   [�߂Asݞ�   As�yP   As��P   [��As��   As��P   As�[P   [�p�As��   As�[P   As��P   [��lAs���   As��P   As�=P   [�s�As�b�   As�=P   As�P   [؂kAs���   As�P   As�P   [�.�As�D�   As�P   As�P   [�d�As��   As�P   As�P   [�2�As�&�   As�P   As�rP   [�K�As��   As�rP   As��P   [�x�As��   As��P   As�TP   [�wAs�y�   As�TP   As��P   [�m�As���   As��P   As�6P   [�g/As�[�   As�6P   As��P   [ب"As���   As��P   AtP   [�%`At=�   AtP   At�P   [��qAt��   At�P   At�P   [�ZAt�   At�P   At	kP   [��r