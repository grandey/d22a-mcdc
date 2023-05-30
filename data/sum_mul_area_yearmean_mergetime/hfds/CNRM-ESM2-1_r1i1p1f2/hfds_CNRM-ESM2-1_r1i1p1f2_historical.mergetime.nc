CDF   �   
      time       bnds      lon       lat          6   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-09-15T06:43:37Z   description       CMIP6 historical   title         =CNRM-ESM2-1 model output prepared for CMIP6 / CMIP historical      activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.13   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Vhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.historical.none.r1i1p1f2     grid      2native ocean tri-polar grid with 105 k ocean cells     
grid_label        gn     nominal_resolution        100 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         ocean      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Omon   variable_id       hfds   variant_label         r1i1p1f2   EXPID         "CNRM-ESM2-1_historical_r1i1p1f2_v2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          92ddb3d0d8ce79f498d792fc8e559dcf   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      history      �Tue May 30 16:59:07 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.hfds.gn.v20181206/hfds_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.yearmean.mul.areacello_historical_v20181206.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CNRM-ESM2-1_r1i1p1f2/hfds_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc
Thu Nov 03 22:36:35 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.hfds.gn.v20181206/hfds_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacello/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Ofx.areacello.gn.v20181206/areacello_Ofx_CNRM-ESM2-1_historical_r1i1p1f2_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.hfds.gn.v20181206/hfds_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.yearmean.mul.areacello_historical_v20181206.fldsum.nc
none     tracking_id       1hdl:21.14100/ef40b25a-ba66-4f9b-9224-8ada09c62a1e      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                      standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    online_operation      average    interval_operation        1800 s     interval_write        1 month    description       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any "flux adjustment") .    positive      down   history       none   cell_measures         area: areacello             �eacello             �                Aq���   Aq��P   Aq�P   Ն��Aq�6�   Aq�P   Aq��P   ִ��Aq���   Aq��P   Aq��P   �ynAq��   Aq��P   Aq�dP   W?��Aq���   Aq�dP   Aq��P   ��!�Aq���   Aq��P   Aq�FP   W��\Aq�k�   Aq�FP   Aq��P   թ:Aq���   Aq��P   Aq�(P   �$7JAq�M�   Aq�(P   Aq��P   V���Aq���   Aq��P   Aq�
P   �I*Aq�/�   Aq�
P   Aq�{P   �
�nAq���   Aq�{P   Aq��P   V#�Aq��   Aq��P   Aq�]P   ծl%AqĂ�   Aq�]P   Aq��P   ֥.oAq���   Aq��P   Aq�?P   �1oAq�d�   Aq�?P   Aq˰P   WVd�Aq���   Aq˰P   Aq�!P   VM�jAq�F�   Aq�!P   AqВP   ֱ3{Aqз�   AqВP   Aq�P   V�-\Aq�(�   Aq�P   Aq�tP   VR�[Aqՙ�   Aq�tP   Aq��P   W.�.Aq�
�   Aq��P   Aq�VP   ���Aq�{�   Aq�VP   Aq��P   Vq�Aq���   Aq��P   Aq�8P   �@SAq�]�   Aq�8P   Aq�P   �^��Aq���   Aq�P   Aq�P   WH�Aq�?�   Aq�P   Aq�P   V@Aq��   Aq�P   Aq��P   �@��Aq�!�   Aq��P   Aq�mP   ��Aq��   Aq�mP   Aq��P   W"��Aq��   Aq��P   Aq�OP   V�"�Aq�t�   Aq�OP   Aq��P   �1�XAq���   Aq��P   Aq�1P   ե��Aq�V�   Aq�1P   Aq��P   �Ҕ7Aq���   Aq��P   Aq�P   ��H�Aq�8�   Aq�P   Aq��P   պ�Aq���   Aq��P   Aq��P   V8�Aq��   Aq��P   ArfP   WB��Ar��   ArfP   Ar�P   ��]�Ar��   Ar�P   ArHP   W�XArm�   ArHP   Ar�P   ՉIfAr��   Ar�P   Ar*P   ��Q�ArO�   Ar*P   Ar�P   WİcAr��   Ar�P   ArP   U�l�Ar1�   ArP   Ar}P   ֻ��Ar��   Ar}P   Ar�P   V&}�Ar�   Ar�P   Ar_P   W#��Ar��   Ar_P   Ar�P   V?��Ar��   Ar�P   ArAP   ԛE�Arf�   ArAP   Ar�P   Vd�Ar��   Ar�P   Ar!#P   Wr��Ar!H�   Ar!#P   Ar#�P   �j��Ar#��   Ar#�P   Ar&P   VVAr&*�   Ar&P   Ar(vP   ִZ�Ar(��   Ar(vP   Ar*�P   �,XgAr+�   Ar*�P   Ar-XP   W�n�Ar-}�   Ar-XP   Ar/�P   W�l�Ar/��   Ar/�P   Ar2:P   �T<�Ar2_�   Ar2:P   Ar4�P   W�Ar4��   Ar4�P   Ar7P   VZJ;Ar7A�   Ar7P   Ar9�P   Wv�RAr9��   Ar9�P   Ar;�P   V~C�Ar<#�   Ar;�P   Ar>oP   ����Ar>��   Ar>oP   Ar@�P   V���ArA�   Ar@�P   ArCQP   V���ArCv�   ArCQP   ArE�P   V�nArE��   ArE�P   ArH3P   W�ArHX�   ArH3P   ArJ�P   Ud1ArJ��   ArJ�P   ArMP   Ws9�ArM:�   ArMP   ArO�P   V�x�ArO��   ArO�P   ArQ�P   V�-zArR�   ArQ�P   ArThP   V���ArT��   ArThP   ArV�P   Uϼ~ArV��   ArV�P   ArYJP   Ջ5�ArYo�   ArYJP   Ar[�P   Wj��Ar[��   Ar[�P   Ar^,P   WS2)Ar^Q�   Ar^,P   Ar`�P   We{�Ar`��   Ar`�P   ArcP   W�Arc3�   ArcP   AreP   V]�<Are��   AreP   Arg�P   W2;Arh�   Arg�P   ArjaP   W0�Arj��   ArjaP   Arl�P   Vd�vArl��   Arl�P   AroCP   V\�Aroh�   AroCP   Arq�P   W��hArq��   Arq�P   Art%P   U몠ArtJ�   Art%P   Arv�P   V�u�Arv��   Arv�P   AryP   W�gAry,�   AryP   Ar{xP   Vp��Ar{��   Ar{xP   Ar}�P   U8��Ar~�   Ar}�P   Ar�ZP   V��gAr��   Ar�ZP   Ar��P   WvO�Ar���   Ar��P   Ar�<P   V�*Ar�a�   Ar�<P   Ar��P   V�Ar���   Ar��P   Ar�P   W�E�Ar�C�   Ar�P   Ar��P   V�(�Ar���   Ar��P   Ar� P   WP(aAr�%�   Ar� P   Ar�qP   WL�Ar���   Ar�qP   Ar��P   V�CAr��   Ar��P   Ar�SP   V���Ar�x�   Ar�SP   Ar��P   W�$Ar���   Ar��P   Ar�5P   Wp]Ar�Z�   Ar�5P   Ar��P   Ըr�Ar���   Ar��P   Ar�P   WP,Ar�<�   Ar�P   Ar��P   W�Ar���   Ar��P   Ar��P   V�D�Ar��   Ar��P   Ar�jP   ���Ar���   Ar�jP   Ar��P   W�7&Ar� �   Ar��P   Ar�LP   U���Ar�q�   Ar�LP   Ar��P   V���Ar���   Ar��P   Ar�.P   W?zAr�S�   Ar�.P   Ar��P   V��>Ar���   Ar��P   Ar�P   V M^Ar�5�   Ar�P   Ar��P   ֱ�NAr���   Ar��P   Ar��P   �"�tAr��   Ar��P   Ar�cP   ��U�Ar���   Ar�cP   Ar��P   W��Ar���   Ar��P   Ar�EP   U�~~Ar�j�   Ar�EP   ArĶP   �$�3Ar���   ArĶP   Ar�'P   W�b�Ar�L�   Ar�'P   ArɘP   ֽ�^Arɽ�   ArɘP   Ar�	P   W	WAr�.�   Ar�	P   Ar�zP   �U�ArΟ�   Ar�zP   Ar��P   W\�Ar��   Ar��P   Ar�\P   ����ArӁ�   Ar�\P   Ar��P   W��Ar���   Ar��P   Ar�>P   W/�Ar�c�   Ar�>P   ArگP   W 0�Ar���   ArگP   Ar� P   W�Ar�E�   Ar� P   ArߑP   V��XAr߶�   ArߑP   Ar�P   Wk�!Ar�'�   Ar�P   Ar�sP   W���Ar��   Ar�sP   Ar��P   �5�Ar�	�   Ar��P   Ar�UP   W?�NAr�z�   Ar�UP   Ar��P   W9[�Ar���   Ar��P   Ar�7P   V��-Ar�\�   Ar�7P   Ar�P   V�FAr���   Ar�P   Ar�P   V˓OAr�>�   Ar�P   Ar��P   W��"Ar���   Ar��P   Ar��P   W\jAr� �   Ar��P   Ar�lP   W���Ar���   Ar�lP   Ar��P   WŮAr��   Ar��P   Ar�NP   �'�fAr�s�   Ar�NP   As�P   �P:As��   As�P   As0P   V�JzAsU�   As0P   As�P   W�	�As��   As�P   As	P   W~��As	7�   As	P   As�P   W��As��   As�P   As�P   W�#As�   As�P   AseP   WW�{As��   AseP   As�P   Wi�zAs��   As�P   AsGP   W7z�Asl�   AsGP   As�P   V�_�As��   As�P   As)P   W���AsN�   As)P   As�P   W%AAs��   As�P   AsP   V禧As0�   AsP   As!|P   WkoTAs!��   As!|P   As#�P   W��	As$�   As#�P   As&^P   W5P�As&��   As&^P   As(�P   W<2�As(��   As(�P   As+@P   W�
wAs+e�   As+@P   As-�P   W�CAs-��   As-�P   As0"P   WM�As0G�   As0"P   As2�P   W��!As2��   As2�P   As5P   W�	�As5)�   As5P   As7uP   W�W: