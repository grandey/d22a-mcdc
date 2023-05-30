CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       oACCESS-ESM1.5 (2019): 
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
grid_label        gn     history      �Tue May 30 16:59:04 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-ESM1-5_r1i1p1f1/hfds_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Omon.hfds.gn.v20210318/hfds_Omon_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacello_ssp126_v20210318.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Omon.hfds.gn.v20210318/hfds_Omon_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_210101-230012.yearmean.mul.areacello_ssp126_v20210318.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-ESM1-5_r1i1p1f1/hfds_ACCESS-ESM1-5_r1i1p1f1_ssp126.mergetime.nc
Tue May 30 16:59:04 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Omon.hfds.gn.v20191115/hfds_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20191115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-ESM1-5_r1i1p1f1/hfds_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc
Thu Nov 03 22:05:50 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Omon.hfds.gn.v20191115/hfds_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacello/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Ofx.areacello.gn.v20191115/areacello_Ofx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Omon.hfds.gn.v20191115/hfds_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20191115.fldsum.nc
2019-11-15T16:13:51Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        CSIRO      mip_era       CMIP6      nominal_resolution        250 km     notes         �Exp: ESM-historical; Local ID: HI-05; Variable: hfds (['sfc_hflux_from_runoff', 'sfc_hflux_coupler', 'sfc_hflux_from_water_evap', 'sfc_hflux_from_water_prec', 'frazil_2d'])   parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      ACCESS-ESM1-5      parent_time_units         days since 0101-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         ACCESS-ESM1-5      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         'ACCESS-ESM1-5 output prepared for CMIP6    variable_id       hfds   variant_label         r1i1p1f1   version       	v20191115      cmor_version      3.4.0      tracking_id       1hdl:21.14100/325696e9-595a-4daf-b492-b54792ca3a57      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                     lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       o2019-11-15T16:13:49Z altered by CMOR: replaced missing value flag (-1e+20) with standard missing value (1e+20).                             Aq���   Aq��P   Aq�P   ����Aq�6�   Aq�P   Aq��P   ֟�Aq���   Aq��P   Aq��P   W&Z�Aq��   Aq��P   Aq�dP   V&�BAq���   Aq�dP   Aq��P   օ�OAq���   Aq��P   Aq�FP   T�U�Aq�k�   Aq�FP   Aq��P   ֚Z�Aq���   Aq��P   Aq�(P   Ձ�NAq�M�   Aq�(P   Aq��P   Ӱ�Aq���   Aq��P   Aq�
P   WN��Aq�/�   Aq�
P   Aq�{P   W��Aq���   Aq�{P   Aq��P   �vuAq��   Aq��P   Aq�]P   �B!�AqĂ�   Aq�]P   Aq��P   U���Aq���   Aq��P   Aq�?P   �NSAq�d�   Aq�?P   Aq˰P   V�cDAq���   Aq˰P   Aq�!P   V�� Aq�F�   Aq�!P   AqВP   VAqз�   AqВP   Aq�P   ��,Aq�(�   Aq�P   Aq�tP   �ۤ*Aqՙ�   Aq�tP   Aq��P   V�0Aq�
�   Aq��P   Aq�VP   W#�Aq�{�   Aq�VP   Aq��P   V���Aq���   Aq��P   Aq�8P   VH�Aq�]�   Aq�8P   Aq�P   V��(Aq���   Aq�P   Aq�P   �f��Aq�?�   Aq�P   Aq�P   V���Aq��   Aq�P   Aq��P   W�Aq�!�   Aq��P   Aq�mP   ֞��Aq��   Aq�mP   Aq��P   UA'3Aq��   Aq��P   Aq�OP   ���;Aq�t�   Aq�OP   Aq��P   WT%Aq���   Aq��P   Aq�1P   W
 �Aq�V�   Aq�1P   Aq��P   ׊��Aq���   Aq��P   Aq�P   ��Aq�8�   Aq�P   Aq��P   �f��Aq���   Aq��P   Aq��P   V&&{Aq��   Aq��P   ArfP   ���Ar��   ArfP   Ar�P   ��}Ar��   Ar�P   ArHP   W�xArm�   ArHP   Ar�P   ��B?Ar��   Ar�P   Ar*P   մެArO�   Ar*P   Ar�P   �H-Ar��   Ar�P   ArP   V���Ar1�   ArP   Ar}P   W=�5Ar��   Ar}P   Ar�P   W��Ar�   Ar�P   Ar_P   UT8Ar��   Ar_P   Ar�P   W�7�Ar��   Ar�P   ArAP   V6�:Arf�   ArAP   Ar�P   UdL�Ar��   Ar�P   Ar!#P   օ2[Ar!H�   Ar!#P   Ar#�P   �l��Ar#��   Ar#�P   Ar&P   �:�tAr&*�   Ar&P   Ar(vP   �)��Ar(��   Ar(vP   Ar*�P   �C�Ar+�   Ar*�P   Ar-XP   WKآAr-}�   Ar-XP   Ar/�P   V�\�Ar/��   Ar/�P   Ar2:P   U��"Ar2_�   Ar2:P   Ar4�P   U䕞Ar4��   Ar4�P   Ar7P   U���Ar7A�   Ar7P   Ar9�P   W��Ar9��   Ar9�P   Ar;�P   ֧��Ar<#�   Ar;�P   Ar>oP   ׸��Ar>��   Ar>oP   Ar@�P   �AS|ArA�   Ar@�P   ArCQP   U�l"ArCv�   ArCQP   ArE�P   V:۾ArE��   ArE�P   ArH3P   ���-ArHX�   ArH3P   ArJ�P   �2�ArJ��   ArJ�P   ArMP   �	
ArM:�   ArMP   ArO�P   �!Q�ArO��   ArO�P   ArQ�P   Wc�ArR�   ArQ�P   ArThP   ��:OArT��   ArThP   ArV�P   ��)ArV��   ArV�P   ArYJP   WN�!ArYo�   ArYJP   Ar[�P   V��FAr[��   Ar[�P   Ar^,P   �إ�Ar^Q�   Ar^,P   Ar`�P   ֆ�Ar`��   Ar`�P   ArcP   V��~Arc3�   ArcP   AreP   ՜��Are��   AreP   Arg�P   W.S�Arh�   Arg�P   ArjaP   V��Arj��   ArjaP   Arl�P   V5w/Arl��   Arl�P   AroCP   �?��Aroh�   AroCP   Arq�P   U�[XArq��   Arq�P   Art%P   V�M�ArtJ�   Art%P   Arv�P   ֮LArv��   Arv�P   AryP   U0��Ary,�   AryP   Ar{xP   Vv5Ar{��   Ar{xP   Ar}�P   V�n�Ar~�   Ar}�P   Ar�ZP   W5�Ar��   Ar�ZP   Ar��P   ViFAr���   Ar��P   Ar�<P   ֑_Ar�a�   Ar�<P   Ar��P   V�-Ar���   Ar��P   Ar�P   V��1Ar�C�   Ar�P   Ar��P   ���QAr���   Ar��P   Ar� P   V"�Ar�%�   Ar� P   Ar�qP   ��Ar���   Ar�qP   Ar��P   V�8Ar��   Ar��P   Ar�SP   U�7Ar�x�   Ar�SP   Ar��P   V�o�Ar���   Ar��P   Ar�5P   W'��Ar�Z�   Ar�5P   Ar��P   V��Ar���   Ar��P   Ar�P   �&��Ar�<�   Ar�P   Ar��P   V�"Ar���   Ar��P   Ar��P   W7��Ar��   Ar��P   Ar�jP   ��Ar���   Ar�jP   Ar��P   U��Ar� �   Ar��P   Ar�LP   V��Ar�q�   Ar�LP   Ar��P   V�xiAr���   Ar��P   Ar�.P   R�]�Ar�S�   Ar�.P   Ar��P   �:qAr���   Ar��P   Ar�P   W�|�Ar�5�   Ar�P   Ar��P   �@��Ar���   Ar��P   Ar��P   ���Ar��   Ar��P   Ar�cP   ׹�0Ar���   Ar�cP   Ar��P   �IbAr���   Ar��P   Ar�EP   ��L�Ar�j�   Ar�EP   ArĶP   ձW&Ar���   ArĶP   Ar�'P   W��Ar�L�   Ar�'P   ArɘP   Ք�=Arɽ�   ArɘP   Ar�	P   V�]+Ar�.�   Ar�	P   Ar�zP   ���ArΟ�   Ar�zP   Ar��P   V��Ar��   Ar��P   Ar�\P   T8q�ArӁ�   Ar�\P   Ar��P   �3aWAr���   Ar��P   Ar�>P   V��dAr�c�   Ar�>P   ArگP   V�fAr���   ArگP   Ar� P   V?�Ar�E�   Ar� P   ArߑP   V�8�Ar߶�   ArߑP   Ar�P   Wu �Ar�'�   Ar�P   Ar�sP   V��	Ar��   Ar�sP   Ar��P   V�yAr�	�   Ar��P   Ar�UP   Vu��Ar�z�   Ar�UP   Ar��P   �tԼAr���   Ar��P   Ar�7P   �FA�Ar�\�   Ar�7P   Ar�P   ՝C�Ar���   Ar�P   Ar�P   W�j�Ar�>�   Ar�P   Ar��P   V�4HAr���   Ar��P   Ar��P   Wv=�Ar� �   Ar��P   Ar�lP   V��Ar���   Ar�lP   Ar��P   V��Ar��   Ar��P   Ar�NP   ֠´Ar�s�   Ar�NP   As�P   �i�As��   As�P   As0P   �3��AsU�   As0P   As�P   W6V�As��   As�P   As	P   W�BkAs	7�   As	P   As�P   W�t�As��   As�P   As�P   W�n�As�   As�P   AseP   Wb��As��   AseP   As�P   W+	�As��   As�P   AsGP   W���Asl�   AsGP   As�P   W>[iAs��   As�P   As)P   V~*[AsN�   As)P   As�P   WZ�DAs��   As�P   AsP   W��As0�   AsP   As!|P   W\�As!��   As!|P   As#�P   Wa�As$�   As#�P   As&^P   W�aAs&��   As&^P   As(�P   W�o�As(��   As(�P   As+@P   W|=uAs+e�   As+@P   As-�P   W�+As-��   As-�P   As0"P   W�ntAs0G�   As0"P   As2�P   W��]As2��   As2�P   As5P   W�{�As5)�   As5P   As7uP   W�p�As7��   As7uP   As9�P   W<��As:�   As9�P   As<WP   W�OxAs<|�   As<WP   As>�P   Wҳ�As>��   As>�P   AsA9P   X��AsA^�   AsA9P   AsC�P   X��AsC��   AsC�P   AsFP   W�G`AsF@�   AsFP   AsH�P   W���AsH��   AsH�P   AsJ�P   W��AsK"�   AsJ�P   AsMnP   XEVAsM��   AsMnP   AsO�P   XL8AsP�   AsO�P   AsRPP   W��;AsRu�   AsRPP   AsT�P   W���AsT��   AsT�P   AsW2P   X�1AsWW�   AsW2P   AsY�P   W磍AsY��   AsY�P   As\P   X�VAs\9�   As\P   As^�P   W��PAs^��   As^�P   As`�P   W��tAsa�   As`�P   AscgP   W��aAsc��   AscgP   Ase�P   W�;Ase��   Ase�P   AshIP   X��Ashn�   AshIP   Asj�P   W�qAsj��   Asj�P   Asm+P   X<�AsmP�   Asm+P   Aso�P   X�Aso��   Aso�P   AsrP   X#xAsr2�   AsrP   Ast~P   W���Ast��   Ast~P   Asv�P   X��Asw�   Asv�P   Asy`P   X�^Asy��   Asy`P   As{�P   X��As{��   As{�P   As~BP   W��As~g�   As~BP   As��P   W��+As���   As��P   As�$P   Xi�As�I�   As�$P   As��P   W�)�As���   As��P   As�P   X+�As�+�   As�P   As�wP   X'%As���   As�wP   As��P   X,��As��   As��P   As�YP   W��rAs�~�   As�YP   As��P   W�g�As���   As��P   As�;P   W��xAs�`�   As�;P   As��P   X��As���   As��P   As�P   W�+�As�B�   As�P   As��P   X�OAs���   As��P   As��P   X1�UAs�$�   As��P   As�pP   X��As���   As�pP   As��P   W�_As��   As��P   As�RP   W��1As�w�   As�RP   As��P   Wۂ=As���   As��P   As�4P   X2As�Y�   As�4P   As��P   X�*As���   As��P   As�P   X�As�;�   As�P   As��P   W�M�As���   As��P   As��P   X �As��   As��P   As�iP   X+{�As���   As�iP   As��P   W���As���   As��P   As�KP   X�bAs�p�   As�KP   As��P   W��QAs���   As��P   As�-P   W�oBAs�R�   As�-P   AsP   W��As���   AsP   As�P   WǑ�As�4�   As�P   AsǀP   X1oAsǥ�   AsǀP   As��P   WՊ�As��   As��P   As�bP   W�v�Aṡ�   As�bP   As��P   W��As���   As��P   As�DP   W�As�i�   As�DP   AsӵP   W̛�As���   AsӵP   As�&P   W�(�As�K�   As�&P   AsؗP   W�T)Asؼ�   AsؗP   As�P   W�_As�-�   As�P   As�yP   W��Asݞ�   As�yP   As��P   X 9�As��   As��P   As�[P   X�hAs��   As�[P   As��P   W�Y�As���   As��P   As�=P   W^�9As�b�   As�=P   As�P   W��sAs���   As�P   As�P   W���As�D�   As�P   As�P   W�qAAs��   As�P   As�P   W��As�&�   As�P   As�rP   W�'As��   As�rP   As��P   We\�As��   As��P   As�TP   WϹ�As�y�   As�TP   As��P   W��As���   As��P   As�6P   W�As�[�   As�6P   As��P   W0�As���   As��P   AtP   W�R�At=�   AtP   At�P   Wܖ�At��   At�P   At�P   W��At�   At�P   At	kP   W�3�At	��   At	kP   At�P   W���At�   At�P   AtMP   W�}7Atr�   AtMP   At�P   W�At��   At�P   At/P   WĚ�AtT�   At/P   At�P   WMZhAt��   At�P   AtP   Wê�At6�   AtP   At�P   W˺�At��   At�P   At�P   W��5At�   At�P   AtdP   W�ϨAt��   AtdP   At!�P   W�OkAt!��   At!�P   At$FP   V��At$k�   At$FP   At&�P   W���At&��   At&�P   At)(P   WiSAt)M�   At)(P   At+�P   W�!�At+��   At+�P   At.
P   W���At./�   At.
P   At0{P   W�o�At0��   At0{P   At2�P   W�PJAt3�   At2�P   At5]P   W�At5��   At5]P   At7�P   W�VFAt7��   At7�P   At:?P   X�eAt:d�   At:?P   At<�P   W�?At<��   At<�P   At?!P   W���At?F�   At?!P   AtA�P   WX�XAtA��   AtA�P   AtDP   W�$%AtD(�   AtDP   AtFtP   W�G�AtF��   AtFtP   AtH�P   Vh�kAtI
�   AtH�P   AtKVP   W@�YAtK{�   AtKVP   AtM�P   WT+�AtM��   AtM�P   AtP8P   W-"�AtP]�   AtP8P   AtR�P   W��HAtR��   AtR�P   AtUP   W$�FAtU?�   AtUP   AtW�P   W�0�AtW��   AtW�P   AtY�P   V��fAtZ!�   AtY�P   At\mP   W�r�At\��   At\mP   At^�P   WeK�At_�   At^�P   AtaOP   WD^Atat�   AtaOP   Atc�P   W4��Atc��   Atc�P   Atf1P   W�aAtfV�   Atf1P   Ath�P   W�x@Ath��   Ath�P   AtkP   W�Q�Atk8�   AtkP   Atm�P   W1�Atm��   Atm�P   Ato�P   WrRrAtp�   Ato�P   AtrfP   W_��Atr��   AtrfP   Att�P   W*I�Att��   Att�P   AtwHP   W��VAtwm�   AtwHP   Aty�P   W��QAty��   Aty�P   At|*P   W��At|O�   At|*P   At~�P   W���At~��   At~�P   At�P   WC:At�1�   At�P   At�}P   WsUAt���   At�}P   At��P   W���At��   At��P   At�_P   W�c.At���   At�_P   At��P   W�g�At���   At��P   At�AP   W�M{At�f�   At�AP   At��P   W6�xAt���   At��P   At�#P   W�U(At�H�   At�#P   At��P   W� zAt���   At��P   At�P   W�KlAt�*�   At�P   At�vP   WP.1At���   At�vP   At��P   W[��At��   At��P   At�XP   W�cAt�}�   At�XP   At��P   W��"At���   At��P   At�:P   Vg�At�_�   At�:P   At��P   W��QAt���   At��P   At�P   V�>�At�A�   At�P   At��P   W��At���   At��P   At��P   W��At�#�   At��P   At�oP   X	H�At���   At�oP   At��P   W6�At��   At��P   At�QP   W��At�v�   At�QP   At��P   W��/At���   At��P   At�3P   W�	�At�X�   At�3P   At��P   Wf��At���   At��P   At�P   W ��At�:�   At�P   At��P   We�eAt���   At��P   At��P   W��dAt��   At��P   At�hP   We
�Atō�   At�hP   At��P   W�TAt���   At��P   At�JP   W��!At�o�   At�JP   At̻P   V�g?At���   At̻P   At�,P   W�/CAt�Q�   At�,P   AtѝP   We��At���   AtѝP   At�P   WK1�At�3�   At�P   At�P   W��:At֤�   At�P   At��P   W*�At��   At��P   At�aP   Wj�/Atۆ�   At�aP   At��P   V�^�At���   At��P   At�CP   Vn�At�h�   At�CP   At�P   W�LnAt���   At�P   At�%P   WA�At�J�   At�%P   At�P   W9��At��   At�P   At�P   W�>�At�,�   At�P   At�xP   X��At��   At�xP   At��P   W�At��   At��P   At�ZP   W��At��   At�ZP   At��P   W�k�At���   At��P   At�<P   W�jtAt�a�   At�<P   At��P   WU|1At���   At��P   At�P   W"OAt�C�   At�P   At��P   V� At���   At��P   Au  P   Հ��Au %�   Au  P   AuqP   W,kAu��   AuqP   Au�P   W�;sAu�   Au�P   AuSP   WEz�Aux�   AuSP   Au	�P   W�1�Au	��   Au	�P   Au5P   WV?PAuZ�   Au5P   Au�P   W�1Au��   Au�P   AuP   W���Au<�   AuP   Au�P   V԰Au��   Au�P   Au�P   W���Au�   Au�P   AujP   W�NAu��   AujP   Au�P   W��Au �   Au�P   AuLP   V���Auq�   AuLP   Au�P   V��!Au��   Au�P   Au".P   W�aAu"S�   Au".P   Au$�P   W��uAu$��   Au$�P   Au'P   V�DAu'5�   Au'P   Au)�P   W�Au)��   Au)�P   Au+�P   W;��Au,�   Au+�P   Au.cP   V��Au.��   Au.cP   Au0�P   W��Au0��   Au0�P   Au3EP   W�z�Au3j�   Au3EP   Au5�P   W�<�Au5��   Au5�P   Au8'P   W�Au8L�   Au8'P   Au:�P   U�_Au:��   Au:�P   Au=	P   V���Au=.�   Au=	P   Au?zP   W$FAu?��   Au?zP   AuA�P   W/۩AuB�   AuA�P   AuD\P   WT��AuD��   AuD\P   AuF�P   W_X�AuF��   AuF�P   AuI>P   W*��AuIc�   AuI>P   AuK�P   W*��AuK��   AuK�P   AuN P   W�ZfAuNE�   AuN P   AuP�P   Ӭ�,AuP��   AuP�P   AuSP   W��AuS'�   AuSP   AuUsP   W��AuU��   AuUsP   AuW�P   V�g�AuX	�   AuW�P   AuZUP   WF��AuZz�   AuZUP   Au\�P   WN��Au\��   Au\�P   Au_7P   W�b=Au_\�   Au_7P   Aua�P   V�9�Aua��   Aua�P   AudP   VWAud>�   AudP   Auf�P   W�ǘAuf��   Auf�P   Auh�P   W�C�Aui �   Auh�P   AuklP   W�b�Auk��   AuklP   Aum�P   WN:2Aun�   Aum�P   AupNP   Wg�1Aups�   AupNP   Aur�P   W!�Aur��   Aur�P   Auu0P   W���AuuU�   Auu0P   Auw�P   W^�Auw��   Auw�P   AuzP   Wf��Auz7�   AuzP   Au|�P   Vd�(Au|��   Au|�P   Au~�P   W3�Au�   Au~�P   Au�eP   W1�Au���   Au�eP   Au��P   W�f�Au���   Au��P   Au�GP   V�z�Au�l�   Au�GP   Au��P   V|4�Au���   Au��P   Au�)P   Wy�Au�N�   Au�)P   Au��P   W]C.Au���   Au��P   Au�P   WY��Au�0�   Au�P   Au�|P   V��zAu���   Au�|P   Au��P   WT��Au��   Au��P   Au�^P   W��Au���   Au�^P   Au��P   W3k�Au���   Au��P   Au�@P   W��Au�e�   Au�@P   Au��P   W%	BAu���   Au��P   Au�"P   W�R�Au�G�   Au�"P   Au��P   WeW#Au���   Au��P   Au�P   W���Au�)�   Au�P   Au�uP   V��#Au���   Au�uP   Au��P   V�]Au��   Au��P   Au�WP   Wd�Au�|�   Au�WP   Au��P   V�I�Au���   Au��P   Au�9P   V�T�Au�^�   Au�9P   Au��P   V��Au���   Au��P   Au�P   Vvo�Au�@�   Au�P   Au��P   W�Au���   Au��P   Au��P   Wl��Au�"�   Au��P   Au�nP   Wd�Au���   Au�nP   Au��P   V?ٺAu��   Au��P   Au�PP   W�uAu�u�   Au�PP   Au��P   Wy�Au���   Au��P   Au�2P   W;JAu�W�   Au�2P   AuʣP   V���Au���   AuʣP   Au�P   Vd]�Au�9�   Au�P   AuυP   W�Z�AuϪ�   AuυP   Au��P   Wx;RAu��   Au��P   Au�gP   W�J�AuԌ�   Au�gP   Au��P   W)�5Au���   Au��P   Au�IP   V�[�Au�n�   Au�IP   AuۺP   �BvAu���   AuۺP   Au�+P   W��RAu�P�   Au�+P   Au��P   W)5KAu���   Au��P   Au�P   Vta�Au�2�   Au�P   Au�~P   T�[�Au��   Au�~P   Au��P   W��Au��   Au��P   Au�`P   W�(Au��   Au�`P   Au��P   W�P9Au���   Au��P   Au�BP   W�&�Au�g�   Au�BP   Au�P   W�]