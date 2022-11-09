CDF   �   
      time       bnds      lon       lat          8   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-CM6-1 (2017):  aerosol: prescribed monthly fields computed by TACTIC_v2 scheme atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: OZL_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) seaIce: Gelato 6.1     institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    name      �/scratch/work/voldoire/outputs/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/18500101/hfds_Omon_CNRM-CM6-1_historical_r1i1p1f2_gn_%start_date%-%end_date%      creation_date         2018-06-20T08:39:51Z   description       CMIP6 historical   title         <CNRM-CM6-1 model output prepared for CMIP6 / CMIP historical   activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.10   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-CM6-1.historical.none.r1i1p1f2      grid      2native ocean tri-polar grid with 105 k ocean cells     
grid_label        gn     nominal_resolution        100 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      
CNRM-CM6-1     parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         ocean      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         
CNRM-CM6-1     source_type       AOGCM      sub_experiment_id         none   sub_experiment        none   table_id      Omon   variable_id       hfds   variant_label         r1i1p1f2   EXPID         CNRM-CM6-1_historical_r1i1p1f2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          d6225e658d7de0912fca2a4293dbe2a7   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      tracking_id       1hdl:21.14100/66ffa017-2990-4146-90f6-809785d52ade      history      5Wed Nov 09 19:01:33 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CNRM-CM6-1_r1i1p1f2/hfds_CNRM-CM6-1_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CNRM-CM6-1_r1i1p1f2/CMIP6.ScenarioMIP.CNRM-CERFACS.CNRM-CM6-1.ssp245.r1i1p1f2.Omon.hfds.gn.v20190219/hfds_Omon_CNRM-CM6-1_ssp245_r1i1p1f2_gn_201501-210012.yearmean.mul.areacello_ssp245_v20190219.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CNRM-CM6-1_r1i1p1f2/hfds_CNRM-CM6-1_r1i1p1f2_ssp245.mergetime.nc
Wed Nov 09 19:01:33 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Omon.hfds.gn.v20180917/hfds_Omon_CNRM-CM6-1_historical_r1i1p1f2_gn_185001-201412.yearmean.mul.areacello_historical_v20180917.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CNRM-CM6-1_r1i1p1f2/hfds_CNRM-CM6-1_r1i1p1f2_historical.mergetime.nc
Thu Nov 03 22:35:54 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Omon.hfds.gn.v20180917/hfds_Omon_CNRM-CM6-1_historical_r1i1p1f2_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacello/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Ofx.areacello.gn.v20180917/areacello_Ofx_CNRM-CM6-1_historical_r1i1p1f2_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CNRM-CM6-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Omon.hfds.gn.v20180917/hfds_Omon_CNRM-CM6-1_historical_r1i1p1f2_gn_185001-201412.yearmean.mul.areacello_historical_v20180917.fldsum.nc
Mon Jul 30 08:05:11 2018: ncatted -O -a tracking_id,global,m,c,hdl:21.14100/66ffa017-2990-4146-90f6-809785d52ade /scratch/work/voldoire/outputs/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/assembled/hfds_Omon_CNRM-CM6-1_historical_r1i1p1f2_gn_185001-201412.nc
none      NCO       "4.5.5"    CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                      standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    online_operation      average    interval_operation        1800 s     interval_write        1 month    description       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any "flux adjustment") .    positive      down   history       none   cell_measures         area: areacello             �                Aq���   Aq��P   Aq�P   ��?Aq�6�   Aq�P   Aq��P   V�$�Aq���   Aq��P   Aq��P   V�/ZAq��   Aq��P   Aq�dP   V�Aq���   Aq�dP   Aq��P   ��zAq���   Aq��P   Aq�FP   W*:NAq�k�   Aq�FP   Aq��P   V��|Aq���   Aq��P   Aq�(P   ���Aq�M�   Aq�(P   Aq��P   �zTuAq���   Aq��P   Aq�
P   �VT�Aq�/�   Aq�
P   Aq�{P   V��+Aq���   Aq�{P   Aq��P   V��pAq��   Aq��P   Aq�]P   V�,�AqĂ�   Aq�]P   Aq��P   V�!Aq���   Aq��P   Aq�?P   �:�YAq�d�   Aq�?P   Aq˰P   V��Aq���   Aq˰P   Aq�!P   ���Aq�F�   Aq�!P   AqВP   V��Aqз�   AqВP   Aq�P   V��EAq�(�   Aq�P   Aq�tP   V��bAqՙ�   Aq�tP   Aq��P   �cokAq�
�   Aq��P   Aq�VP   ��{Aq�{�   Aq�VP   Aq��P   V���Aq���   Aq��P   Aq�8P   V�I�Aq�]�   Aq�8P   Aq�P   U�g@Aq���   Aq�P   Aq�P   WK�nAq�?�   Aq�P   Aq�P   V���Aq��   Aq�P   Aq��P   ��8�Aq�!�   Aq��P   Aq�mP   V�k�Aq��   Aq�mP   Aq��P   Վ��Aq��   Aq��P   Aq�OP   �$?Aq�t�   Aq�OP   Aq��P   W�,AAq���   Aq��P   Aq�1P   �7�9Aq�V�   Aq�1P   Aq��P   ױ�Aq���   Aq��P   Aq�P   ֹQtAq�8�   Aq�P   Aq��P   �i^�Aq���   Aq��P   Aq��P   ���OAq��   Aq��P   ArfP   W>�Ar��   ArfP   Ar�P   � ��Ar��   Ar�P   ArHP   Wb�Arm�   ArHP   Ar�P   V�Ar��   Ar�P   Ar*P   V�sArO�   Ar*P   Ar�P   W8��Ar��   Ar�P   ArP   �&�Ar1�   ArP   Ar}P   �ӣAr��   Ar}P   Ar�P   W��Ar�   Ar�P   Ar_P   W6TAr��   Ar_P   Ar�P   �PB�Ar��   Ar�P   ArAP   U}k�Arf�   ArAP   Ar�P   W��Ar��   Ar�P   Ar!#P   �
[Ar!H�   Ar!#P   Ar#�P   TT��Ar#��   Ar#�P   Ar&P   V���Ar&*�   Ar&P   Ar(vP   ��Ar(��   Ar(vP   Ar*�P   ��!�Ar+�   Ar*�P   Ar-XP   W#�^Ar-}�   Ar-XP   Ar/�P   W�Ar/��   Ar/�P   Ar2:P   �\.qAr2_�   Ar2:P   Ar4�P   Wo Ar4��   Ar4�P   Ar7P   WOAr7A�   Ar7P   Ar9�P   V���Ar9��   Ar9�P   Ar;�P   �E,�Ar<#�   Ar;�P   Ar>oP   V��Ar>��   Ar>oP   Ar@�P   ���ArA�   Ar@�P   ArCQP   V��bArCv�   ArCQP   ArE�P   W��SArE��   ArE�P   ArH3P   WV�ArHX�   ArH3P   ArJ�P   VA�ArJ��   ArJ�P   ArMP   W�OArM:�   ArMP   ArO�P   V���ArO��   ArO�P   ArQ�P   U�wBArR�   ArQ�P   ArThP   Vu#ArT��   ArThP   ArV�P   V�WBArV��   ArV�P   ArYJP   W=MArYo�   ArYJP   Ar[�P   V��Ar[��   Ar[�P   Ar^,P   ���iAr^Q�   Ar^,P   Ar`�P   W&�Ar`��   Ar`�P   ArcP   Vߎ}Arc3�   ArcP   AreP   Wl'uAre��   AreP   Arg�P   Wc��Arh�   Arg�P   ArjaP   VH�Arj��   ArjaP   Arl�P   Vͦ	Arl��   Arl�P   AroCP   V.�Aroh�   AroCP   Arq�P   W�Arq��   Arq�P   Art%P   V�/1ArtJ�   Art%P   Arv�P   Vٳ�Arv��   Arv�P   AryP   W�9�Ary,�   AryP   Ar{xP   W�B2Ar{��   Ar{xP   Ar}�P   W6JAr~�   Ar}�P   Ar�ZP   W�PAr��   Ar�ZP   Ar��P   WK_pAr���   Ar��P   Ar�<P   W!)�Ar�a�   Ar�<P   Ar��P   ���Ar���   Ar��P   Ar�P   V�KAr�C�   Ar�P   Ar��P   W�Ar���   Ar��P   Ar� P   �a�EAr�%�   Ar� P   Ar�qP   WC�|Ar���   Ar�qP   Ar��P   W��Ar��   Ar��P   Ar�SP   WH@kAr�x�   Ar�SP   Ar��P   ֛�Ar���   Ar��P   Ar�5P   V�jlAr�Z�   Ar�5P   Ar��P   Ue�Ar���   Ar��P   Ar�P   �(zRAr�<�   Ar�P   Ar��P   W~�Ar���   Ar��P   Ar��P   V��qAr��   Ar��P   Ar�jP   V���Ar���   Ar�jP   Ar��P   Wd2#Ar� �   Ar��P   Ar�LP   W�K�Ar�q�   Ar�LP   Ar��P   �L�	Ar���   Ar��P   Ar�.P   Wi@�Ar�S�   Ar�.P   Ar��P   W�wAr���   Ar��P   Ar�P   �p�.Ar�5�   Ar�P   Ar��P   V�"Ar���   Ar��P   Ar��P   ք}`Ar��   Ar��P   Ar�cP   ���Ar���   Ar�cP   Ar��P   ��`�Ar���   Ar��P   Ar�EP   V>�Ar�j�   Ar�EP   ArĶP   W�Ar���   ArĶP   Ar�'P   V�/dAr�L�   Ar�'P   ArɘP   ֈgArɽ�   ArɘP   Ar�	P   WO�Ar�.�   Ar�	P   Ar�zP   VhhArΟ�   Ar�zP   Ar��P   V&�Ar��   Ar��P   Ar�\P   Vt�_ArӁ�   Ar�\P   Ar��P   Wb"�Ar���   Ar��P   Ar�>P   ֱ��Ar�c�   Ar�>P   ArگP   W���Ar���   ArگP   Ar� P   ՞��Ar�E�   Ar� P   ArߑP   WnsjAr߶�   ArߑP   Ar�P   W50Ar�'�   Ar�P   Ar�sP   �J��Ar��   Ar�sP   Ar��P   WPAr�	�   Ar��P   Ar�UP   �X�Ar�z�   Ar�UP   Ar��P   W2��Ar���   Ar��P   Ar�7P   V�v�Ar�\�   Ar�7P   Ar�P   Ut|�Ar���   Ar�P   Ar�P   V��Ar�>�   Ar�P   Ar��P   UʙAr���   Ar��P   Ar��P   V�itAr� �   Ar��P   Ar�lP   W���Ar���   Ar�lP   Ar��P   W2ΚAr��   Ar��P   Ar�NP   W��Ar�s�   Ar�NP   As�P   ֘Q�As��   As�P   As0P   �|�AsU�   As0P   As�P   WF�As��   As�P   As	P   V�uAs	7�   As	P   As�P   W��jAs��   As�P   As�P   W�As�   As�P   AseP   Wz��As��   AseP   As�P   W:pPAs��   As�P   AsGP   WV1Asl�   AsGP   As�P   VvҭAs��   As�P   As)P   W�M�AsN�   As)P   As�P   Wz0As��   As�P   AsP   WN6As0�   AsP   As!|P   W���As!��   As!|P   As#�P   W�[pAs$�   As#�P   As&^P   V�q�As&��   As&^P   As(�P   W\�As(��   As(�P   As+@P   W��As+e�   As+@P   As-�P   W��As-��   As-�P   As0"P   W�n�As0G�   As0"P   As2�P   W��EAs2��   As2�P   As5P   W�ۿAs5)�   As5P   As7uP   W�:HAs7��   As7uP   As9�P   W��>As:�   As9�P   As<WP   W�Y�As<|�   As<WP   As>�P   Wص�As>��   As>�P   AsA9P   W���AsA^�   AsA9P   AsC�P   W�i�AsC��   AsC�P   AsFP   W��AsF@�   AsFP   AsH�P   W���AsH��   AsH�P   AsJ�P   W��AAsK"�   AsJ�P   AsMnP   W��AsM��   AsMnP   AsO�P   W��]AsP�   AsO�P   AsRPP   W�<AsRu�   AsRPP   AsT�P   X& �AsT��   AsT�P   AsW2P   X@�AsWW�   AsW2P   AsY�P   W��CAsY��   AsY�P   As\P   W�wAs\9�   As\P   As^�P   X7��As^��   As^�P   As`�P   W�d4Asa�   As`�P   AscgP   X�Asc��   AscgP   Ase�P   W��IAse��   Ase�P   AshIP   X �Ashn�   AshIP   Asj�P   X9�)Asj��   Asj�P   Asm+P   X�AsmP�   Asm+P   Aso�P   X?GAso��   Aso�P   AsrP   XR�Asr2�   AsrP   Ast~P   W���Ast��   Ast~P   Asv�P   XnAsw�   Asv�P   Asy`P   X
�
Asy��   Asy`P   As{�P   W�	|As{��   As{�P   As~BP   X�xAs~g�   As~BP   As��P   W�'�As���   As��P   As�$P   X� As�I�   As�$P   As��P   X3�EAs���   As��P   As�P   X��As�+�   As�P   As�wP   X6��As���   As�wP   As��P   X*0�As��   As��P   As�YP   W��As�~�   As�YP   As��P   X"�As���   As��P   As�;P   X ��As�`�   As�;P   As��P   X#��As���   As��P   As�P   X��As�B�   As�P   As��P   XD
~As���   As��P   As��P   X)�"As�$�   As��P   As�pP   X#D As���   As�pP   As��P   X5�EAs��   As��P   As�RP   X(��As�w�   As�RP   As��P   X&�?As���   As��P   As�4P   X-��As�Y�   As�4P   As��P   Xk6�As���   As��P   As�P   X
e�As�;�   As�P   As��P   X0�;As���   As��P   As��P   X/��As��   As��P   As�iP   X#3zAs���   As�iP   As��P   X;ʹAs���   As��P   As�KP   X"��As�p�   As�KP   As��P   W��As���   As��P   As�-P   XH�>As�R�   As�-P   AsP   W�[oAs���   AsP   As�P   X=ӰAs�4�   As�P   AsǀP   X8a�Asǥ�   AsǀP   As��P   X#T0As��   As��P   As�bP   X%�@Aṡ�   As�bP   As��P   XH%�As���   As��P   As�DP   X;̏As�i�   As�DP   AsӵP   Xb�As���   AsӵP   As�&P   X(tAs�K�   As�&P   AsؗP   X1��Asؼ�   AsؗP   As�P   XW�*As�-�   As�P   As�yP   X�Asݞ�   As�yP   As��P   X\�As��   As��P   As�[P   W���As��   As�[P   As��P   XAY>As���   As��P   As�=P   X.E
As�b�   As�=P   As�P   X'[�As���   As�P   As�P   XB��As�D�   As�P   As�P   X$$vAs��   As�P   As�P   X%�As�&�   As�P   As�rP   X*�UAs��   As�rP   As��P   X ~2As��   As��P   As�TP   XSXEAs�y�   As�TP   As��P   X0��As���   As��P   As�6P   X>�YAs�[�   As�6P   As��P   XHFAs���   As��P   AtP   X(�bAt=�   AtP   At�P   Xf�At��   At�P   At�P   X ��At�   At�P   At	kP   XCzZ