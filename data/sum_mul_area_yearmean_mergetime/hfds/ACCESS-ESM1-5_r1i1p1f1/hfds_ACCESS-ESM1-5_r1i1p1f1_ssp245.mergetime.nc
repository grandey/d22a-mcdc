CDF   �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       oACCESS-ESM1.5 (2019): 
aerosol: CLASSIC (v1.0)
atmos: HadGAM2 (r1.1, N96; 192 x 145 longitude/latitude; 38 levels; top level 39255 m)
atmosChem: none
land: CABLE2.4
landIce: none
ocean: ACCESS-OM2 (MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: WOMBAT (same grid as ocean)
seaIce: CICE4.1 (same grid as ocean)    institution       aCommonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         @�f�       creation_date         2019-11-15T16:13:51Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Qhttps://furtherinfo.es-doc.org/CMIP6.CSIRO.ACCESS-ESM1-5.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (145x192 latxlon)   
grid_label        gn     history      �Wed Nov 09 19:01:25 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-ESM1-5_r1i1p1f1/hfds_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp245.r1i1p1f1.Omon.hfds.gn.v20191115/hfds_Omon_ACCESS-ESM1-5_ssp245_r1i1p1f1_gn_201501-210012.yearmean.mul.areacello_ssp245_v20191115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-ESM1-5_r1i1p1f1/hfds_ACCESS-ESM1-5_r1i1p1f1_ssp245.mergetime.nc
Wed Nov 09 19:01:24 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Omon.hfds.gn.v20191115/hfds_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20191115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-ESM1-5_r1i1p1f1/hfds_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc
Thu Nov 03 22:05:50 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Omon.hfds.gn.v20191115/hfds_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacello/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Ofx.areacello.gn.v20191115/areacello_Ofx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Omon.hfds.gn.v20191115/hfds_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20191115.fldsum.nc
2019-11-15T16:13:51Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.    initialization_index            institution_id        CSIRO      mip_era       CMIP6      nominal_resolution        250 km     notes         �Exp: ESM-historical; Local ID: HI-05; Variable: hfds (['sfc_hflux_from_runoff', 'sfc_hflux_coupler', 'sfc_hflux_from_water_evap', 'sfc_hflux_from_water_prec', 'frazil_2d'])   parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      ACCESS-ESM1-5      parent_time_units         days since 0101-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         ACCESS-ESM1-5      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         'ACCESS-ESM1-5 output prepared for CMIP6    variable_id       hfds   variant_label         r1i1p1f1   version       	v20191115      cmor_version      3.4.0      tracking_id       1hdl:21.14100/325696e9-595a-4daf-b492-b54792ca3a57      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       o2019-11-15T16:13:49Z altered by CMOR: replaced missing value flag (-1e+20) with standard missing value (1e+20).                             Aq���   Aq��P   Aq�P   ����Aq�6�   Aq�P   Aq��P   ֟�Aq���   Aq��P   Aq��P   W&Z�Aq��   Aq��P   Aq�dP   V&�BAq���   Aq�dP   Aq��P   օ�OAq���   Aq��P   Aq�FP   T�U�Aq�k�   Aq�FP   Aq��P   ֚Z�Aq���   Aq��P   Aq�(P   Ձ�NAq�M�   Aq�(P   Aq��P   Ӱ�Aq���   Aq��P   Aq�
P   WN��Aq�/�   Aq�
P   Aq�{P   W��Aq���   Aq�{P   Aq��P   �vuAq��   Aq��P   Aq�]P   �B!�AqĂ�   Aq�]P   Aq��P   U���Aq���   Aq��P   Aq�?P   �NSAq�d�   Aq�?P   Aq˰P   V�cDAq���   Aq˰P   Aq�!P   V�� Aq�F�   Aq�!P   AqВP   VAqз�   AqВP   Aq�P   ��,Aq�(�   Aq�P   Aq�tP   �ۤ*Aqՙ�   Aq�tP   Aq��P   V�0Aq�
�   Aq��P   Aq�VP   W#�Aq�{�   Aq�VP   Aq��P   V���Aq���   Aq��P   Aq�8P   VH�Aq�]�   Aq�8P   Aq�P   V��(Aq���   Aq�P   Aq�P   �f��Aq�?�   Aq�P   Aq�P   V���Aq��   Aq�P   Aq��P   W�Aq�!�   Aq��P   Aq�mP   ֞��Aq��   Aq�mP   Aq��P   UA'3Aq��   Aq��P   Aq�OP   ���;Aq�t�   Aq�OP   Aq��P   WT%Aq���   Aq��P   Aq�1P   W
 �Aq�V�   Aq�1P   Aq��P   ׊��Aq���   Aq��P   Aq�P   ��Aq�8�   Aq�P   Aq��P   �f��Aq���   Aq��P   Aq��P   V&&{Aq��   Aq��P   ArfP   ���Ar��   ArfP   Ar�P   ��}Ar��   Ar�P   ArHP   W�xArm�   ArHP   Ar�P   ��B?Ar��   Ar�P   Ar*P   մެArO�   Ar*P   Ar�P   �H-Ar��   Ar�P   ArP   V���Ar1�   ArP   Ar}P   W=�5Ar��   Ar}P   Ar�P   W��Ar�   Ar�P   Ar_P   UT8Ar��   Ar_P   Ar�P   W�7�Ar��   Ar�P   ArAP   V6�:Arf�   ArAP   Ar�P   UdL�Ar��   Ar�P   Ar!#P   օ2[Ar!H�   Ar!#P   Ar#�P   �l��Ar#��   Ar#�P   Ar&P   �:�tAr&*�   Ar&P   Ar(vP   �)��Ar(��   Ar(vP   Ar*�P   �C�Ar+�   Ar*�P   Ar-XP   WKآAr-}�   Ar-XP   Ar/�P   V�\�Ar/��   Ar/�P   Ar2:P   U��"Ar2_�   Ar2:P   Ar4�P   U䕞Ar4��   Ar4�P   Ar7P   U���Ar7A�   Ar7P   Ar9�P   W��Ar9��   Ar9�P   Ar;�P   ֧��Ar<#�   Ar;�P   Ar>oP   ׸��Ar>��   Ar>oP   Ar@�P   �AS|ArA�   Ar@�P   ArCQP   U�l"ArCv�   ArCQP   ArE�P   V:۾ArE��   ArE�P   ArH3P   ���-ArHX�   ArH3P   ArJ�P   �2�ArJ��   ArJ�P   ArMP   �	
ArM:�   ArMP   ArO�P   �!Q�ArO��   ArO�P   ArQ�P   Wc�ArR�   ArQ�P   ArThP   ��:OArT��   ArThP   ArV�P   ��)ArV��   ArV�P   ArYJP   WN�!ArYo�   ArYJP   Ar[�P   V��FAr[��   Ar[�P   Ar^,P   �إ�Ar^Q�   Ar^,P   Ar`�P   ֆ�Ar`��   Ar`�P   ArcP   V��~Arc3�   ArcP   AreP   ՜��Are��   AreP   Arg�P   W.S�Arh�   Arg�P   ArjaP   V��Arj��   ArjaP   Arl�P   V5w/Arl��   Arl�P   AroCP   �?��Aroh�   AroCP   Arq�P   U�[XArq��   Arq�P   Art%P   V�M�ArtJ�   Art%P   Arv�P   ֮LArv��   Arv�P   AryP   U0��Ary,�   AryP   Ar{xP   Vv5Ar{��   Ar{xP   Ar}�P   V�n�Ar~�   Ar}�P   Ar�ZP   W5�Ar��   Ar�ZP   Ar��P   ViFAr���   Ar��P   Ar�<P   ֑_Ar�a�   Ar�<P   Ar��P   V�-Ar���   Ar��P   Ar�P   V��1Ar�C�   Ar�P   Ar��P   ���QAr���   Ar��P   Ar� P   V"�Ar�%�   Ar� P   Ar�qP   ��Ar���   Ar�qP   Ar��P   V�8Ar��   Ar��P   Ar�SP   U�7Ar�x�   Ar�SP   Ar��P   V�o�Ar���   Ar��P   Ar�5P   W'��Ar�Z�   Ar�5P   Ar��P   V��Ar���   Ar��P   Ar�P   �&��Ar�<�   Ar�P   Ar��P   V�"Ar���   Ar��P   Ar��P   W7��Ar��   Ar��P   Ar�jP   ��Ar���   Ar�jP   Ar��P   U��Ar� �   Ar��P   Ar�LP   V��Ar�q�   Ar�LP   Ar��P   V�xiAr���   Ar��P   Ar�.P   R�]�Ar�S�   Ar�.P   Ar��P   �:qAr���   Ar��P   Ar�P   W�|�Ar�5�   Ar�P   Ar��P   �@��Ar���   Ar��P   Ar��P   ���Ar��   Ar��P   Ar�cP   ׹�0Ar���   Ar�cP   Ar��P   �IbAr���   Ar��P   Ar�EP   ��L�Ar�j�   Ar�EP   ArĶP   ձW&Ar���   ArĶP   Ar�'P   W��Ar�L�   Ar�'P   ArɘP   Ք�=Arɽ�   ArɘP   Ar�	P   V�]+Ar�.�   Ar�	P   Ar�zP   ���ArΟ�   Ar�zP   Ar��P   V��Ar��   Ar��P   Ar�\P   T8q�ArӁ�   Ar�\P   Ar��P   �3aWAr���   Ar��P   Ar�>P   V��dAr�c�   Ar�>P   ArگP   V�fAr���   ArگP   Ar� P   V?�Ar�E�   Ar� P   ArߑP   V�8�Ar߶�   ArߑP   Ar�P   Wu �Ar�'�   Ar�P   Ar�sP   V��	Ar��   Ar�sP   Ar��P   V�yAr�	�   Ar��P   Ar�UP   Vu��Ar�z�   Ar�UP   Ar��P   �tԼAr���   Ar��P   Ar�7P   �FA�Ar�\�   Ar�7P   Ar�P   ՝C�Ar���   Ar�P   Ar�P   W�j�Ar�>�   Ar�P   Ar��P   V�4HAr���   Ar��P   Ar��P   Wv=�Ar� �   Ar��P   Ar�lP   V��Ar���   Ar�lP   Ar��P   V��Ar��   Ar��P   Ar�NP   ֠´Ar�s�   Ar�NP   As�P   �i�As��   As�P   As0P   �3��AsU�   As0P   As�P   W6V�As��   As�P   As	P   W�BkAs	7�   As	P   As�P   W�t�As��   As�P   As�P   W�n�As�   As�P   AseP   Wb��As��   AseP   As�P   W+	�As��   As�P   AsGP   W���Asl�   AsGP   As�P   W>[iAs��   As�P   As)P   V~*[AsN�   As)P   As�P   WZ�DAs��   As�P   AsP   W��As0�   AsP   As!|P   W\�As!��   As!|P   As#�P   Wa�As$�   As#�P   As&^P   W�aAs&��   As&^P   As(�P   W�o�As(��   As(�P   As+@P   W|=uAs+e�   As+@P   As-�P   W�+As-��   As-�P   As0"P   W�ntAs0G�   As0"P   As2�P   W��]As2��   As2�P   As5P   W�{�As5)�   As5P   As7uP   W�p�As7��   As7uP   As9�P   W�NAs:�   As9�P   As<WP   W�\As<|�   As<WP   As>�P   W��As>��   As>�P   AsA9P   W�nCAsA^�   AsA9P   AsC�P   W�ُAsC��   AsC�P   AsFP   X3`|AsF@�   AsFP   AsH�P   XF�AsH��   AsH�P   AsJ�P   W㙞AsK"�   AsJ�P   AsMnP   X |�AsM��   AsMnP   AsO�P   W�{�AsP�   AsO�P   AsRPP   X)&AsRu�   AsRPP   AsT�P   W�q8AsT��   AsT�P   AsW2P   X��AsWW�   AsW2P   AsY�P   W�VPAsY��   AsY�P   As\P   W�2WAs\9�   As\P   As^�P   W�O�As^��   As^�P   As`�P   X/�4Asa�   As`�P   AscgP   XRdAsc��   AscgP   Ase�P   X��Ase��   Ase�P   AshIP   W�Ashn�   AshIP   Asj�P   W�o�Asj��   Asj�P   Asm+P   X(r�AsmP�   Asm+P   Aso�P   X��Aso��   Aso�P   AsrP   XI�Asr2�   AsrP   Ast~P   XZ�Ast��   Ast~P   Asv�P   W�QAsw�   Asv�P   Asy`P   X2��Asy��   Asy`P   As{�P   X��As{��   As{�P   As~BP   X�TAs~g�   As~BP   As��P   XN{As���   As��P   As�$P   XDeaAs�I�   As�$P   As��P   W�pAs���   As��P   As�P   X*��As�+�   As�P   As�wP   X]<As���   As�wP   As��P   X��As��   As��P   As�YP   X��As�~�   As�YP   As��P   W֦As���   As��P   As�;P   W�D*As�`�   As�;P   As��P   X,`�As���   As��P   As�P   X0�As�B�   As�P   As��P   X�As���   As��P   As��P   X&r�As�$�   As��P   As�pP   XQAs���   As�pP   As��P   X$�As��   As��P   As�RP   X)r�As�w�   As�RP   As��P   XXAs���   As��P   As�4P   X1�7As�Y�   As�4P   As��P   X)4�As���   As��P   As�P   X4�As�;�   As�P   As��P   X�7As���   As��P   As��P   X@�3As��   As��P   As�iP   X?�As���   As�iP   As��P   X}tAs���   As��P   As�KP   XMϖAs�p�   As�KP   As��P   X*qRAs���   As��P   As�-P   XNa~As�R�   As�-P   AsP   W�XAs���   AsP   As�P   X.��As�4�   As�P   AsǀP   XY'�Asǥ�   AsǀP   As��P   XR[vAs��   As��P   As�bP   X"�lAṡ�   As�bP   As��P   X;�As���   As��P   As�DP   XL�'As�i�   As�DP   AsӵP   X�As���   AsӵP   As�&P   XK8!As�K�   As�&P   AsؗP   X#�LAsؼ�   AsؗP   As�P   X�9As�-�   As�P   As�yP   X$	Asݞ�   As�yP   As��P   X&THAs��   As��P   As�[P   X�QAs��   As�[P   As��P   X>��As���   As��P   As�=P   X)�wAs�b�   As�=P   As�P   X3DjAs���   As�P   As�P   XJ02As�D�   As�P   As�P   X>6As��   As�P   As�P   X:�As�&�   As�P   As�rP   X9�As��   As�rP   As��P   X$�As��   As��P   As�TP   XG�^As�y�   As�TP   As��P   XU	KAs���   As��P   As�6P   XA� As�[�   As�6P   As��P   W� �As���   As��P   AtP   X%�.At=�   AtP   At�P   X=�gAt��   At�P   At�P   XK�cAt�   At�P   At	kP   X �K